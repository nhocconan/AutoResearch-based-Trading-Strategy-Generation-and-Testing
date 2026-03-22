#!/usr/bin/env python3
"""
Experiment #575: 12h KAMA Adaptive Trend with Dual HTF HMA Bias + Fisher Reversals

Hypothesis: After 500+ failed experiments, the key insight is:
1. KAMA (Kaufman Adaptive MA) adapts to volatility - works better than fixed EMA/HMA in crypto
2. Dual HTF bias (1d + 1w HMA) provides stronger trend confirmation than single HTF
3. Fisher Transform catches reversals at extremes without whipsaw
4. 12h timeframe = fewer trades, less fee drag, captures multi-day crypto trends
5. Looser ADX>15 (not >25) ensures we generate enough trades (>10 per symbol)
6. Conservative sizing (0.30) protects against 2022-style crashes

Why this should work on 12h:
- KAMA efficiency ratio adapts to ranging vs trending markets automatically
- 1d + 1w HMA both aligned = strong trend confirmation (both must agree)
- Fisher Transform < -1.5 for long, > +1.5 for short = proven reversal levels
- ADX>15 is loose enough to trigger but filters worst chop
- 2.5*ATR stoploss protects capital during crashes
- Discrete signal levels (0.0, ±0.30) minimize fee churn

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d + 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_dual_htf_hma_fisher_reversal_adx_atr_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=period, min_periods=period).sum()
    er = signal / noise.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for clearer reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    price = (high_s + low_s) / 2
    
    # Highest high and lowest low over period
    highest = price.rolling(window=period, min_periods=period).max()
    lowest = price.rolling(window=period, min_periods=period).min()
    
    # Normalize price
    x = 0.67 * (price - lowest) / (highest - lowest).replace(0, np.inf) - 0.33
    x = x.clip(-0.99, 0.99)  # Prevent log domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x).replace(0, np.inf))
    fisher = fisher.fillna(0)
    
    return fisher.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10)
    fisher_9 = calculate_fisher_transform(high, low, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(kama_10[i]) or np.isnan(fisher_9[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF HMA TREND BIAS (both must agree) ===
        bull_bias_1d = close[i] > hma_1d_aligned[i]
        bull_bias_1w = close[i] > hma_1w_aligned[i]
        bear_bias_1d = close[i] < hma_1d_aligned[i]
        bear_bias_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bull: both 1d and 1w HMA bullish
        strong_bull = bull_bias_1d and bull_bias_1w
        # Strong bear: both 1d and 1w HMA bearish
        strong_bear = bear_bias_1d and bear_bias_1w
        
        # === KAMA TREND DIRECTION ===
        # KAMA sloping up = bullish, sloping down = bearish
        kama_bull = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_bear = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher < -1.5 = oversold (long signal)
        # Fisher > +1.5 = overbought (short signal)
        fisher_oversold = fisher_9[i] < -1.5
        fisher_overbought = fisher_9[i] > 1.5
        
        # Fisher crossing back from extremes
        fisher_cross_up = fisher_9[i] > -1.5 and fisher_9[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher_9[i] < 1.5 and fisher_9[i-1] >= 1.5 if i > 0 else False
        
        # === ADX FILTER (loose threshold for more trades) ===
        trend_present = adx_14[i] > 15  # Loose - ensures we get trades
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: Strong HTF bull bias + KAMA bullish + Fisher oversold/cross + ADX confirms
        if strong_bull and kama_bull and price_above_kama and (fisher_oversold or fisher_cross_up) and trend_present:
            new_signal = SIZE
        
        # Short: Strong HTF bear bias + KAMA bearish + Fisher overbought/cross + ADX confirms
        elif strong_bear and kama_bear and price_below_kama and (fisher_overbought or fisher_cross_down) and trend_present:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if dual HTF bias flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        # Exit if KAMA slope flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bear:
                new_signal = 0.0
            if position_side < 0 and kama_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals