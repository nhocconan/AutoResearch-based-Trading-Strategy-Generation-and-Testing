#!/usr/bin/env python3
"""
Experiment #407: 12h KAMA Adaptive Trend + 1d/1w HMA Bias + RSI Pullback + Chop Filter

Hypothesis: After 406 experiments, the key insight is ADAPTIVE indicators + REGIME filtering.
KAMA (Kaufman Adaptive Moving Average) adjusts smoothing based on market efficiency ratio,
performing better than fixed EMA/HMA in mixed regimes (2021 bull, 2022 crash, 2023-24 range, 2025 bear).

STRATEGY COMPONENTS:
1. KAMA(21) on 12h: Adaptive trend following - smoother in chop, responsive in trends
   - ER (Efficiency Ratio) = |close - close[n]| / sum(|close[i] - close[i-1]|)
   - SC (Smoothing Constant) = ER * (fast_sc - slow_sc) + slow_sc
   - KAMA = KAMA_prev + SC * (close - KAMA_prev)

2. 1d HMA(21) + 1w HMA(40): Dual HTF trend bias
   - Long only when price > both 1d and 1w HMA (strong bull)
   - Short only when price < both 1d and 1w HMA (strong bear)
   - Flat when mixed signals (avoid whipsaw)

3. RSI(14) Pullback Entries: Better timing than breakouts
   - Long: price > KAMA + RSI crosses above 40 from below (pullback end)
   - Short: price < KAMA + RSI crosses below 60 from above (rally end)
   - Avoids chasing breakouts that reverse

4. Choppiness Index(14) Regime Filter:
   - Only trade when CHOP < 50 (some trend) or CHOP > 65 (clear range)
   - Skip when 50-65 (neutral chop = whipsaw zone)

5. ATR(14) Trailing Stop: 2.5x for risk management
   - Signal → 0 when price moves 2.5*ATR against position

6. Position Sizing: 0.25 discrete (conservative for 12h volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why this should work on 12h:
- 12h captures multi-day trends without intraday noise
- KAMA adapts to 2022 crash (high vol) vs 2023 range (low vol)
- Dual HTF (1d+1w) ensures we only trade with major trend
- RSI pullback = better entry timing than Donchian breakouts
- Should generate 15-40 trades/year per symbol (enough for stats, not too many fees)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_dual_htf_hma_rsi_pullback_chop_atr_v1"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    Smoothing Constant (SC) = ER * (fast_sc - slow_sc) + slow_sc
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    fast_sc = 2/(fast_period+1) = 2/3 for fast response
    slow_sc = 2/(slow_period+1) = 2/31 for slow smoothing
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close - np.roll(close, period))
    price_change[:period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
        if i >= period:
            noise[i] -= np.abs(close[i-period] - close[i-period-1]) if i > period else 0
    
    noise_sum = np.zeros(n)
    for i in range(period, n):
        noise_sum[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    er = np.zeros(n)
    for i in range(period, n):
        if noise_sum[i] > 1e-10:
            er[i] = price_change[i] / noise_sum[i]
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    50-65 = neutral chop (avoid trading)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    hma_1w = calculate_hma(df_1w['close'].values, 40)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, 21, 2, 30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # Long only when price > both 1d and 1w HMA (strong bull)
        # Short only when price < both 1d and 1w HMA (strong bear)
        bull_bias = (close[i] > hma_1d_aligned[i]) and (close[i] > hma_1w_aligned[i])
        bear_bias = (close[i] < hma_1d_aligned[i]) and (close[i] < hma_1w_aligned[i])
        # Mixed/neutral = stay flat
        
        # === CHOPPY INDEX REGIME FILTER ===
        # Only trade when CHOP < 50 (trend) or CHOP > 65 (clear range)
        # Skip when 50-65 (neutral chop = whipsaw zone)
        tradable_regime = (chop[i] < 50) or (chop[i] > 65)
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI crosses above 40 from below (pullback ending in uptrend)
        # Short: RSI crosses below 60 from above (rally ending in downtrend)
        rsi_cross_above_40 = (rsi[i-1] < 40) and (rsi[i] >= 40)
        rsi_cross_below_60 = (rsi[i-1] > 60) and (rsi[i] <= 60)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if tradable_regime:
            # LONG ENTRY: Bull bias + price > KAMA + RSI pullback end
            if bull_bias and kama_bull and rsi_cross_above_40:
                new_signal = SIZE
            
            # SHORT ENTRY: Bear bias + price < KAMA + RSI rally end
            elif bear_bias and kama_bear and rsi_cross_below_60:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if bear bias develops
        if in_position and position_side > 0 and new_signal != 0.0:
            if bear_bias:
                new_signal = 0.0
        
        # Exit short if bull bias develops
        if in_position and position_side < 0 and new_signal != 0.0:
            if bull_bias:
                new_signal = 0.0
        
        # === REGIME EXIT ===
        # Exit if regime becomes non-tradable (neutral chop)
        if in_position and position_side != 0 and new_signal != 0.0:
            if not tradable_regime:
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