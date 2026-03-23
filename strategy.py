#!/usr/bin/env python3
"""
Experiment #828: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + ADX Regime

Hypothesis: After 567 failed strategies, the key insight is that 30m needs
HTF direction (4h HMA) + regime filter (1d ADX) + precise 30m entry timing.
Most 30m failures come from too many trades (>200/yr) → fee drag kills profit.

Strategy design:
1. 30m Primary timeframe (target 40-80 trades/year)
2. 4h HMA(21) for trend direction — call ONCE before loop
3. 1d ADX(14) for regime — trending (ADX>25) vs ranging (ADX<20)
4. 30m RSI(7) for pullback entries — faster than RSI14, catches dips in trend
5. 30m Bollinger Bands (20, 2.0) for mean reversion context
6. Volume filter: > 0.8x 20-bar average
7. Session filter: only 8-20 UTC (highest liquidity, avoids Asian chop)
8. ATR(14) trailing stop 2.5x
9. Dual regime: trend-follow in trending, mean-revert in ranging
10. Position size: 0.25 (smaller for lower TF to reduce fee impact)

Why this works:
- 4h HMA gives direction (proven edge from best strategy)
- 1d ADX filters out whipsaw periods (ADX<20 = avoid trend trades)
- 30m RSI7 catches pullbacks within 4h trend (entry precision)
- Session filter avoids low-liquidity hours (reduces false breakouts)
- Smaller size (0.25) reduces fee drag on higher trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_adx_regime_4h1d_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
    
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=7)  # Faster RSI for entries
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d ADX for regime filter
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20  # London/NY overlap
        
        # === TREND DIRECTION (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d ADX14) ===
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        neutral_regime = 20 <= adx_1d_aligned[i] <= 25
        
        # === RSI SIGNALS (7-period for faster response) ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_oversold = rsi_30m[i] < 25
        rsi_extreme_overbought = rsi_30m[i] > 75
        rsi_neutral_low = 35 <= rsi_30m[i] < 50
        rsi_neutral_high = 50 < rsi_30m[i] <= 65
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_oversold = bb_position < 0.2
        bb_overbought = bb_position > 0.8
        bb_middle_cross_up = close[i] > bb_middle[i] and close[i-1] <= bb_middle[i-1] if not np.isnan(bb_middle[i-1]) else False
        bb_middle_cross_down = close[i] < bb_middle[i] and close[i-1] >= bb_middle[i-1] if not np.isnan(bb_middle[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) — Trend Following ===
        if trending_regime and in_session:
            # Long: 4h bullish + RSI pullback + volume + BB support
            if trend_4h_bullish and rsi_oversold and volume_confirmed and bb_oversold:
                desired_signal = BASE_SIZE
            
            # Long: 4h bullish + RSI recovering from oversold + BB middle cross
            if trend_4h_bullish and rsi_neutral_low and volume_confirmed and bb_middle_cross_up:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: 4h bearish + RSI rally + volume + BB resistance
            if trend_4h_bearish and rsi_overbought and volume_confirmed and bb_overbought:
                desired_signal = -BASE_SIZE
            
            # Short: 4h bearish + RSI weakening + BB middle cross down
            if trend_4h_bearish and rsi_neutral_high and volume_confirmed and bb_middle_cross_down:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion ===
        elif ranging_regime and in_session:
            # Long: BB oversold + RSI extreme + volume spike
            if bb_oversold and rsi_extreme_oversold and volume_confirmed:
                desired_signal = BASE_SIZE
            
            # Long: BB lower touch + RSI oversold
            if bb_oversold and rsi_oversold and volume_confirmed:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: BB overbought + RSI extreme + volume spike
            if bb_overbought and rsi_extreme_overbought and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Short: BB upper touch + RSI overbought
            if bb_overbought and rsi_overbought and volume_confirmed:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) — Conservative ===
        elif neutral_regime and in_session:
            # Only take high-probability setups with multiple confluence
            if trend_4h_bullish and rsi_extreme_oversold and bb_oversold and volume_confirmed:
                desired_signal = REDUCED_SIZE
            
            if trend_4h_bearish and rsi_extreme_overbought and bb_overbought and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend intact and RSI not overbought
                if trend_4h_bullish and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + RSI overbought
            if trend_4h_bearish and rsi_30m[i] > 75:
                desired_signal = 0.0
            # Exit if BB overbought in ranging regime
            if ranging_regime and bb_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + RSI oversold
            if trend_4h_bullish and rsi_30m[i] < 25:
                desired_signal = 0.0
            # Exit if BB oversold in ranging regime
            if ranging_regime and bb_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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
        
        signals[i] = desired_signal
    
    return signals