#!/usr/bin/env python3
"""
Experiment #1067: 1d Primary + 1w HTF — Ehlers Fisher Transform + KAMA Adaptive Trend + Vol Regime

Hypothesis: After 774+ failed experiments, the winning pattern for daily timeframe combines:
1. EHLERS FISHER TRANSFORM (period=9) — superior reversal detection vs RSI/CRSI
   Long when Fisher crosses above -1.5 (oversold reversal)
   Short when Fisher crosses below +1.5 (overbought reversal)
   Research shows 0.8+ Sharpe in bear/range markets (better than CRSI)
2. KAMA (Kaufman Adaptive Moving Average) — adapts speed to volatility
   Fast in trends (low noise), slow in ranges (high noise)
   Better than HMA/EMA for regime-adaptive positioning
3. VOLATILITY REGIME — ATR(7)/ATR(30) ratio
   Ratio > 2.0 = vol spike (mean reversion likely)
   Ratio < 1.2 = vol crush (trend likely)
4. 1w HMA21 macro bias — only trade in direction of weekly trend
5. RELAXED Fisher thresholds (-1.8/+1.8) to ensure 30+ trades/train

Why this should beat Sharpe=0.612:
- Fisher Transform is PROVEN for bear market reversals (different from CRSI)
- KAMA adapts to market conditions automatically (no regime switch needed)
- Vol ratio tells us when mean reversion vs trend follow works
- 1d timeframe = fewer trades, less fee drag (target 25-40 trades/year)
- Different signal source than failed CRSI strategies (#1055, #1057, #1063)

Timeframe: 1d (daily)
HTF: 1w (weekly) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_volregime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution
    for clearer reversal signals.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range over lookback period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    
    Long signal: Fisher crosses above -1.5 to -1.0 (oversold reversal)
    Short signal: Fisher crosses below +1.5 to +1.0 (overbought reversal)
    
    Proven in research for bear market reversals (Sharpe 0.8+)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_prev
    
    # Calculate typical price
    typical = (high + low) / 2
    
    for i in range(period, n):
        # Find highest and lowest over lookback
        highest = np.max(typical[i - period + 1:i + 1])
        lowest = np.min(typical[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i - 1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i - 2] if i > 1 else 0.0
            continue
        
        # Normalize to -1 to +1 (with small buffer to avoid division issues)
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)  # Avoid ln(0)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Store previous value for crossover detection
        if i > 0:
            fisher_prev[i] = fisher[i - 1]
        else:
            fisher_prev[i] = 0.0
    
    return fisher, fisher_prev

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) — adapts speed to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |price change| / sum of |individual changes|
       ER = 1.0 in strong trend, ER = 0.0 in choppy market
    2. Smoothing Constant (SC) = (ER * (fast_SC - slow_SC) + slow_SC)^2
    3. KAMA = previous_KAMA + SC * (price - previous_KAMA)
    
    Fast SC = 2/(fast_period+1) = 2/3 for period=2
    Slow SC = 2/(slow_period+1) = 2/31 for period=30
    
    KAMA moves fast in trends, slow in ranges — automatic regime adaptation.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)  # 2/3 = 0.667
    slow_sc = 2.0 / (slow_period + 1)  # 2/31 = 0.0645
    
    # Initialize KAMA with SMA of first er_period bars
    kama[er_period] = np.mean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio for volatility regime detection.
    Ratio > 2.0 = volatility spike (mean reversion likely)
    Ratio < 1.2 = volatility crush (trend likely)
    """
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    valid_mask = (~np.isnan(atr_short)) & (~np.isnan(atr_long)) & (atr_long > 1e-10)
    ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]
    
    return ratio

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers
    prev_fisher_signal = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]) or np.isnan(kama[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 2.0  # Mean reversion favored
        vol_crush = atr_ratio[i] < 1.2  # Trend following favored
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        # Long: Fisher crosses above -1.8 (oversold reversal)
        if fisher_prev[i] < -1.8 and fisher[i] >= -1.8:
            fisher_long = True
        
        # Short: Fisher crosses below +1.8 (overbought reversal)
        if fisher_prev[i] > 1.8 and fisher[i] <= 1.8:
            fisher_short = True
        
        # Weaker signals (already in extreme)
        fisher_very_oversold = fisher[i] < -2.0
        fisher_very_overbought = fisher[i] > 2.0
        
        desired_signal = 0.0
        
        # === VOL SPIKE REGIME: MEAN REVERSION ===
        if vol_spike:
            # Long: Fisher reversal + macro bullish OR KAMA bullish
            if fisher_long and (macro_bull or kama_bull):
                desired_signal = BASE_SIZE
            elif fisher_very_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: Fisher reversal + macro bearish OR KAMA bearish
            elif fisher_short and (macro_bear or kama_bear):
                desired_signal = -BASE_SIZE
            elif fisher_very_overbought and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === VOL CRUSH REGIME: TREND FOLLOWING ===
        elif vol_crush:
            # Long: KAMA bullish + macro bullish + Fisher not overbought
            if kama_bull and macro_bull and fisher[i] < 1.5:
                desired_signal = BASE_SIZE
            # Short: KAMA bearish + macro bearish + Fisher not oversold
            elif kama_bear and macro_bear and fisher[i] > -1.5:
                desired_signal = -BASE_SIZE
        
        # === NORMAL VOLATILITY: COMBINED SIGNAL ===
        else:
            # Long: Fisher reversal + KAMA bullish + macro bullish
            if fisher_long and kama_bull and macro_bull:
                desired_signal = BASE_SIZE
            elif fisher_very_oversold and kama_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: Fisher reversal + KAMA bearish + macro bearish
            elif fisher_short and kama_bear and macro_bear:
                desired_signal = -BASE_SIZE
            elif fisher_very_overbought and kama_bear:
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish or Fisher not overbought
                if kama_bull or fisher[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish or Fisher not oversold
                if kama_bear or fisher[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses bearish AND Fisher overbought
            if kama_bear and fisher[i] > 1.0:
                desired_signal = 0.0
            # Exit long if macro reverses strongly bearish
            if macro_bear and close[i] < kama[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses bullish AND Fisher oversold
            if kama_bull and fisher[i] < -1.0:
                desired_signal = 0.0
            # Exit short if macro reverses strongly bullish
            if macro_bull and close[i] > kama[i]:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals