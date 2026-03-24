#!/usr/bin/env python3
"""
Experiment #116: 30m Primary + 4h/1d HTF — Fisher Transform + Choppiness + HMA Trend + Session

Hypothesis: After 115 failed experiments, the pattern is clear:
- Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Lower TF (5m/15m/1h) strategies either generate 0 trades OR too many trades (>200/yr)
- 30m is the sweet spot: enough bars for precision, few enough for fee control
- Fisher Transform excels at catching reversals in bear/range markets (proven edge)
- Choppiness Index regime filter prevents trend strategies in choppy markets
- 4h HMA provides trend bias without being too restrictive
- Session filter (08-20 UTC) avoids low-liquidity whipsaws
- LOOSE entry thresholds ensure >=40 trades/year on all symbols

Key design choices:
- Timeframe: 30m (40-80 trades/year target)
- HTF: 4h HMA(21) for trend bias, 1d HMA(50) for major regime
- Entry: Fisher Transform crossover + Choppiness regime + volume confirmation
- Session: 08-20 UTC only (avoid Asian session whipsaws)
- Position size: 0.20 (20% of capital, conservative for 30m)
- Stoploss: 2.5x ATR trailing
- LOOSE Fisher thresholds (-1.2/+1.2) to ensure trades generate

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=40 on train, trades>=4 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_chop_hma_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution, highlights reversals
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest) / range_hl
        # Clamp to avoid log(0)
        price_norm = np.clip(price_norm * 0.99 + 0.005, 0.005, 0.995)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm + 1e-10))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher[i-1] + 0.3 * fisher_val
        
        fisher_prev[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    return vol_ratio

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # Convert ms to seconds, then to datetime
    timestamps = open_time / 1000.0
    # Use pandas to extract hour
    dt = pd.to_datetime(timestamps, unit='s', utc=True)
    hours = dt.hour.values
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Get session hours
    session_hours = get_session_hour(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 30m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        hour = session_hours[i]
        in_session = (hour >= 8) and (hour <= 20)
        
        if not in_session:
            # Outside session: flatten position or hold
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR REGIME (1d HMA) ===
        major_bull = close[i] > hma_1d_aligned[i]
        major_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = choppy/range (mean revert)
        # CHOP <= 55 = trending (trend follow)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # LOOSE thresholds to ensure trades generate
        fisher_cross_up = (fisher_prev[i] < -1.2) and (fisher[i] >= -1.2)
        fisher_cross_down = (fisher_prev[i] > 1.2) and (fisher[i] <= 1.2)
        
        # Extreme Fisher values (strong reversal signals)
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Fisher reversals with HTF bias
            # LONG: Fisher cross up + HTF bull + volume confirmed
            if fisher_cross_up and htf_bull and vol_confirmed:
                desired_signal = SIZE
            # SHORT: Fisher cross down + HTF bear + volume confirmed
            elif fisher_cross_down and htf_bear and vol_confirmed:
                desired_signal = -SIZE
            # Fallback: extreme Fisher with major regime
            elif fisher_extreme_low and major_bull:
                desired_signal = SIZE * 0.7
            elif fisher_extreme_high and major_bear:
                desired_signal = -SIZE * 0.7
        else:
            # CHOPPY REGIME: Mean reversion at Fisher extremes
            # LONG: Fisher extreme low + not strongly bear HTF
            if fisher_extreme_low and not htf_bear:
                desired_signal = SIZE
            # SHORT: Fisher extreme high + not strongly bull HTF
            elif fisher_extreme_high and not htf_bull:
                desired_signal = -SIZE
            # Fallback: Fisher cross with volume
            elif fisher_cross_up and vol_ratio[i] > 1.5:
                desired_signal = SIZE * 0.7
            elif fisher_cross_down and vol_ratio[i] > 1.5:
                desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals