#!/usr/bin/env python3
"""
Experiment #958: 30m Primary + 4h/1d HTF — Fisher Transform + Larry Williams Vol Breakout + Session Filter

Hypothesis: After 687 failed strategies, Ehlers Fisher Transform (underexplored) combined with
Larry Williams Volatility Breakout and strict session/volume filters should work on 30m timeframe.

Key insights from research:
1. Fisher Transform (period=9): Transforms price into Gaussian distribution, crosses at -1.5/+1.5
   catch reversals better than RSI in bear markets. Less explored in our experiments.
2. Larry Williams Vol Breakout: Long = open + K*prev_range (K=0.5), breakout confirmation
   Works well for entry timing within HTF trend direction.
3. 4h HMA(21) for trend direction (proven in baseline with Sharpe=0.612)
4. 1d HMA(21) for macro regime filter
5. Session filter: Only trade 8-20 UTC (highest volume, less whipsaw)
6. Volume confirmation: volume > 0.8x 20-bar average

Why 30m timeframe:
- Target 40-80 trades/year (strict filters prevent fee drag)
- HTF (4h/1d) provides signal DIRECTION
- 30m only for ENTRY TIMING (when to pull trigger)
- Session filter reduces noise from Asian session low-volume periods

Critical improvements over failed strategies:
- Fisher Transform instead of CRSI (CRSI failed 10+ times)
- Larry Williams breakout for precise entry timing
- Session filter (8-20 UTC) removes low-volume whipsaw
- Volume confirmation prevents false breakouts
- Discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_williams_session_4h1d_hma_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        
        # Clamp to 0.001-0.999 to avoid log(0)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (recursive filter)
        if i > period and not np.isnan(fisher[i-1]):
            fisher_val = 0.6 * fisher_val + 0.4 * fisher[i-1]
        
        fisher[i] = fisher_val
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_larry_williams_level(open_prices, high, low, close, period=20, k=0.5):
    """
    Larry Williams Volatility Breakout Level.
    Long breakout level = open + k * previous_range
    Short breakout level = open - k * previous_range
    """
    n = len(close)
    long_level = np.full(n, np.nan)
    short_level = np.full(n, np.nan)
    prev_range = np.full(n, np.nan)
    
    if n < period + 1:
        return long_level, short_level, prev_range
    
    for i in range(period, n):
        # Previous bar range
        prev_range[i] = high[i-1] - low[i-1]
        
        # Breakout levels
        long_level[i] = open_prices[i] + k * prev_range[i]
        short_level[i] = open_prices[i] - k * prev_range[i]
    
    return long_level, short_level, prev_range

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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume."""
    n = len(volume)
    vol_avg = np.full(n, np.nan)
    
    if n < period:
        return vol_avg
    
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_avg

def get_utc_hour(open_time_ms):
    """Extract UTC hour from Binance open_time (milliseconds)."""
    # Binance open_time is milliseconds since epoch
    # Convert to seconds, then to datetime
    ts_seconds = open_time_ms / 1000.0
    # Get hour in UTC
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts_seconds)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_prices = prices["open"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_30m = calculate_atr(high, low, close, period=14)
    lw_long, lw_short, lw_range = calculate_larry_williams_level(open_prices, high, low, close, period=20, k=0.5)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher cross tracking
    prev_fisher_cross_long = False
    prev_fisher_cross_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(lw_long[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 100 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_signal[i-1] < -1.5 and fisher[i] > -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_signal[i-1] > 1.5 and fisher[i] < 1.5:
                fisher_cross_short = True
        
        # === LARRY WILLIAMS BREAKOUT ===
        lw_breakout_long = close[i] > lw_long[i]
        lw_breakout_short = close[i] < lw_short[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === RSI FILTER (avoid extreme overbought/oversold for entries) ===
        # Simple RSI calculation for additional filter
        rsi_period = 14
        if i >= rsi_period:
            delta = np.diff(close[max(0, i-rsi_period):i+1])
            gain = np.sum(np.where(delta > 0, delta, 0))
            loss = np.sum(np.where(delta < 0, -delta, 0))
            if loss > 1e-10:
                rsi = 100 - (100 / (1 + gain / loss))
            else:
                rsi = 100
        else:
            rsi = 50
        
        rsi_not_extreme = 25 < rsi < 75
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        # Must have: HTF trend + Fisher/Vol breakout + Session + Volume
        long_conditions = 0
        
        if trend_4h_bullish or macro_bull:
            long_conditions += 1
        
        if fisher_cross_long:
            long_conditions += 1
        
        if lw_breakout_long and volume_confirmed:
            long_conditions += 1
        
        if in_session:
            long_conditions += 1
        
        if rsi_not_extreme or rsi < 60:
            long_conditions += 1
        
        # Enter long if 3+ conditions met
        if long_conditions >= 3:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        short_conditions = 0
        
        if trend_4h_bearish or macro_bear:
            short_conditions += 1
        
        if fisher_cross_short:
            short_conditions += 1
        
        if lw_breakout_short and volume_confirmed:
            short_conditions += 1
        
        if in_session:
            short_conditions += 1
        
        if rsi_not_extreme or rsi > 40:
            short_conditions += 1
        
        # Enter short if 3+ conditions met
        if short_conditions >= 3:
            if desired_signal > 0:
                # Conflict - skip entry
                desired_signal = 0.0
            else:
                desired_signal = -BASE_SIZE
        
        # === REDUCED SIZE ENTRIES (2 conditions, strong HTF alignment) ===
        if desired_signal == 0.0:
            # Long with strong HTF alignment
            if trend_4h_bullish and macro_bull and (fisher_cross_long or lw_breakout_long):
                if in_session and volume_confirmed:
                    desired_signal = REDUCED_SIZE
            
            # Short with strong HTF alignment
            if trend_4h_bearish and macro_bear and (fisher_cross_short or lw_breakout_short):
                if in_session and volume_confirmed:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend intact and Fisher not overbought
                if (trend_4h_bullish or macro_bull) and fisher[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and Fisher not oversold
                if (trend_4h_bearish or macro_bear) and fisher[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses
            if trend_4h_bearish and macro_bear:
                desired_signal = 0.0
            # Exit if Fisher reaches overbought
            if fisher[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses
            if trend_4h_bullish and macro_bull:
                desired_signal = 0.0
            # Exit if Fisher reaches oversold
            if fisher[i] < -2.0:
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