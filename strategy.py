#!/usr/bin/env python3
"""
Experiment #501: 4h Primary + 1d HTF — KAMA Adaptive Trend + Supertrend + RSI Pullback

Hypothesis: After 449 failed strategies (mostly CRSI/Choppiness/VolSpike combos), try 
a DIFFERENT approach based on proven patterns that haven't been over-tested:

1. KAMA (Kaufman Adaptive Moving Average): Adapts to volatility automatically
   - Fast in trends (ER high), slow in chop (ER low)
   - Reduces whipsaw without needing separate regime filter
   - Only ~5 strategies tried KAMA, most failed due to complex filters
   
2. SUPERTREND (ATR-based): Clean trend direction signal
   - Long when price > Supertrend, Short when price < Supertrend
   - Built-in ATR stop (3.0 multiplier proven optimal)
   - Only 2-3 strategies tried pure Supertrend
   
3. RSI PULLBACK in trend direction (proven pattern from research):
   - Bull trend: enter long on RSI(14) pullback to 40-50
   - Bear trend: enter short on RSI(14) bounce to 50-60
   - Much simpler than CRSI, more trades than extreme RSI
   
4. 1D HMA major trend filter (multi-timeframe confluence):
   - Only take 4h signals in direction of 1d trend
   - Prevents counter-trend trades that fail in 2022 crash

Why this might beat current best (Sharpe=0.435):
- KAMA adapts automatically = no manual regime detection needed
- Supertrend + RSI pullback = proven combo (research note #5)
- Fewer filters = MORE TRADES (critical: >=30/symbol on train)
- 4h TF targets 25-45 trades/year (optimal fee/trade balance)
- Different from 449 failed strategies (no CRSI, no Choppiness, no VolSpike)

Position sizing: 0.28 long, 0.25 short (discrete, max 0.40)
Stoploss: Supertrend flip OR 3.0 * ATR trailing
Target: 25-45 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_supertrend_rsi_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, atr_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs chop).
    
    Efficiency Ratio (ER) = |Close - Close(n)| / Sum(|Close - Close(prev)|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change over period
    change = np.abs(close_s.diff(slow_period).values)
    
    # Sum of absolute changes (volatility)
    abs_diff = np.abs(close_s.diff().values)
    volatility = pd.Series(abs_diff).rolling(window=slow_period, min_periods=slow_period).sum().values
    
    # Efficiency Ratio (0 = noise, 1 = perfect trend)
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[slow_period] = close_s.iloc[slow_period]
    
    for i in range(slow_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    
    # ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend calculation
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    supertrend[period] = upper_band[period]
    
    for i in range(period + 1, n):
        if close[i-1] <= supertrend[i-1]:
            # Was in downtrend
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
        else:
            # Was in uptrend
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
    
    return supertrend, trend

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA for adaptive trend
    kama_40 = calculate_kama(close, fast_period=2, slow_period=30, atr_period=10)
    kama_80 = calculate_kama(close, fast_period=2, slow_period=60, atr_period=10)
    
    # Supertrend for clear trend direction
    supertrend, st_trend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # RSI for pullback entries
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.28
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(kama_40[i]) or np.isnan(supertrend[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Bull: price > 1d HMA(21) AND HMA(21) > HMA(50)
        bull_regime = (close[i] > hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        
        # Bear: price < 1d HMA(21) AND HMA(21) < HMA(50)
        bear_regime = (close[i] < hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        
        # Neutral/range: HMA flat
        neutral_regime = not bull_regime and not bear_regime
        
        # === 4H SUPERTREND DIRECTION ===
        st_bull = st_trend[i] == 1
        st_bear = st_trend[i] == -1
        
        # === 4H KAMA TREND ===
        kama_bull = kama_40[i] > kama_80[i]
        kama_bear = kama_40[i] < kama_80[i]
        
        # === RSI PULLBACK LEVELS ===
        # Long pullback: RSI dropped to 40-50 in uptrend
        rsi_pullback_long = (rsi_14[i] >= 38) and (rsi_14[i] <= 52)
        
        # Short bounce: RSI rose to 50-62 in downtrend
        rsi_bounce_short = (rsi_14[i] >= 48) and (rsi_14[i] <= 62)
        
        # RSI extreme for counter-trend (rare)
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES (confluence of 1d trend + 4h trend + RSI pullback)
        # Condition 1: 1d bull + 4h supertrend bull + RSI pullback (primary signal)
        if bull_regime and st_bull and rsi_pullback_long:
            new_signal = LONG_SIZE
        # Condition 2: 1d bull + KAMA bull + RSI pullback (alternative trend confirm)
        elif bull_regime and kama_bull and rsi_pullback_long:
            new_signal = LONG_SIZE
        # Condition 3: Neutral regime + Supertrend bull + RSI pullback (range breakout)
        elif neutral_regime and st_bull and rsi_pullback_long:
            new_signal = LONG_SIZE * 0.7
        # Condition 4: Extreme RSI oversold (capitulation long, any regime)
        elif rsi_extreme_low and st_bull:
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: 1d bear + 4h supertrend bear + RSI bounce (primary signal)
            if bear_regime and st_bear and rsi_bounce_short:
                new_signal = -SHORT_SIZE
            # Condition 2: 1d bear + KAMA bear + RSI bounce (alternative trend confirm)
            elif bear_regime and kama_bear and rsi_bounce_short:
                new_signal = -SHORT_SIZE
            # Condition 3: Neutral regime + Supertrend bear + RSI bounce (range breakdown)
            elif neutral_regime and st_bear and rsi_bounce_short:
                new_signal = -SHORT_SIZE * 0.7
            # Condition 4: Extreme RSI overbought (FOMO short, any regime)
            elif rsi_extreme_high and st_bear:
                new_signal = -SHORT_SIZE * 0.8
        
        # === STOPLOSS CHECK (Supertrend flip OR 3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            # Supertrend flip to bear
            if st_bear or close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            # Supertrend flip to bull
            if st_bull or close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or profit target) ===
        # Exit long if 1d regime flips bear OR RSI overbought
        if in_position and position_side > 0:
            if bear_regime or rsi_14[i] > 70:
                new_signal = 0.0
        
        # Exit short if 1d regime flips bull OR RSI oversold
        if in_position and position_side < 0:
            if bull_regime or rsi_14[i] < 30:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals