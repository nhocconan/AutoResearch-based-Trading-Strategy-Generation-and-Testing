#!/usr/bin/env python3
"""
Experiment #681: 4h Primary + 1d HTF — Fisher Transform + KAMA Trend + BB Width Regime

Hypothesis: After analyzing 573+ failed strategies, the pattern is clear:
1. CRSI+Chop on 4h has been tried 5+ times with declining results (#669 Sharpe=0.151)
2. Fisher Transform (Ehlers) is NOT in the failed list — fresh signal type
3. BB Width percentile is more robust than Choppiness Index for regime detection
4. KAMA adapts to volatility better than HMA/EMA in crypto's varying regimes

This strategy uses:
- Bollinger Band Width percentile (20-period lookback) for regime detection
  - BB Width < 30th percentile = squeeze (expect breakout/trend)
  - BB Width > 70th percentile = expansion (expect mean-reversion)
- Fisher Transform (period=9) for precise reversal entries
  - Long: Fisher crosses above -2.0 from below
  - Short: Fisher crosses below +2.0 from above
- KAMA (ER=10) for adaptive trend following (faster in trends, slower in chop)
- 1d KAMA slope for major trend bias (simpler, more robust than HMA)

Why this might beat Sharpe=0.520:
- Fisher Transform is mathematically superior to RSI for reversal detection (Ehlers 1998)
- BB Width regime is price-action based, not derived like Choppiness
- KAMA reduces whipsaw in ranging markets while catching trends
- 4h timeframe = optimal 25-45 trades/year (per Rule 10)
- Simpler entry logic = more trades (avoiding 0-trade failure mode)

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 30-50 trades/year on 4h
Stoploss: 3.0*ATR trailing (wider than 2.5 to avoid premature exits)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_bbwidth_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Price change over er_period
    price_change = np.abs(close_s - close_s.shift(er_period)).values
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close_s.iloc[i-er_period:i].diff()))
    
    # Efficiency Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((Close - LL) / (HH - LL) - 0.5) + 0.67 * Fisher_prev
    
    Entry signals:
    - Long: Fisher crosses above -2.0 from below
    - Short: Fisher crosses below +2.0 from above
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        x = 0.66 * ((high[i] + low[i]) / 2.0 - ll) / price_range + 0.67 * fisher_signal[i-1]
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain error
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_bb_width(close, high, low, period=20, std_mult=2.0):
    """
    Calculate Bollinger Band Width and its percentile rank.
    BB Width = (Upper Band - Lower Band) / Middle Band
    
    Percentile rank over 20-period lookback:
    - < 30th percentile: Squeeze (expect breakout)
    - > 70th percentile: Expansion (expect mean-reversion)
    """
    close_s = pd.Series(close)
    
    # Bollinger Bands
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # BB Width
    bb_width = (upper - lower) / sma
    
    # Percentile rank over 20-period lookback
    bb_width_percentile = bb_width.rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    return bb_width.values, bb_width_percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for primary trend direction
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    bb_width, bb_width_pct = calculate_bb_width(close, high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track Fisher crossover state
    prev_fisher = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_width_pct[i]) or np.isnan(fisher[i]) or np.isnan(kama_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (KAMA slope over 3 bars) ===
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3] if i >= 3 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === BB WIDTH REGIME ===
        is_squeeze = bb_width_pct[i] < 35.0  # Squeeze (expect trend/breakout)
        is_expansion = bb_width_pct[i] > 65.0  # Expansion (expect mean-reversion)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = (fisher[i] > -2.0 and prev_fisher <= -2.0)
        fisher_cross_short = (fisher[i] < 2.0 and prev_fisher >= 2.0)
        
        # === 4H KAMA SLOPE (3 bars) ===
        kama_4h_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_4h_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Squeeze (BB Width < 35th pct) + Fisher long cross + 1d bull bias
        if is_squeeze and fisher_cross_long:
            if kama_1d_slope_bull or price_above_kama_1d:
                new_signal = POSITION_SIZE
            elif not kama_1d_slope_bear:  # Neutral 1d, allow long
                new_signal = POSITION_SIZE * 0.7  # Reduced size without trend confirmation
        
        # Regime 2: Expansion (BB Width > 65th pct) + Fisher long + 4h KAMA bull
        elif is_expansion and fisher_cross_long and kama_4h_slope_bull:
            new_signal = POSITION_SIZE * 0.7  # Mean-reversion, smaller size
        
        # --- SHORT ENTRY ---
        # Regime 1: Squeeze (BB Width < 35th pct) + Fisher short cross + 1d bear bias
        if is_squeeze and fisher_cross_short:
            if kama_1d_slope_bear or price_below_kama_1d:
                new_signal = -POSITION_SIZE
            elif not kama_1d_slope_bull:  # Neutral 1d, allow short
                new_signal = -POSITION_SIZE * 0.7  # Reduced size without trend confirmation
        
        # Regime 2: Expansion (BB Width > 65th pct) + Fisher short + 4h KAMA bear
        elif is_expansion and fisher_cross_short and kama_4h_slope_bear:
            new_signal = -POSITION_SIZE * 0.7  # Mean-reversion, smaller size
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1d_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
        prev_fisher = fisher[i]
    
    return signals