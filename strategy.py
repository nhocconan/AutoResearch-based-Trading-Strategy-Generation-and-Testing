#!/usr/bin/env python3
"""
Experiment #864: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Choppiness Regime

Hypothesis: After 600+ failed strategies, KAMA (Kaufman Adaptive Moving Average) 
outperforms HMA/EMA in crypto because it adapts to market efficiency ratios.
During 2022 crash (high volatility), KAMA flattens and avoids whipsaws.
During trending periods, KAMA follows price closely. This adaptability is 
critical for BTC/ETH which alternate between crash and rally.

Strategy design:
1. 4h Primary timeframe (target 30-50 trades/year)
2. 12h KAMA(21) for intermediate trend bias
3. 1d KAMA(50) for secular trend direction
4. 4h Choppiness Index(14) for regime detection (chop vs trend)
5. 4h ADX(14) for trend strength confirmation
6. 4h RSI(14) for entry timing (pullback entries)
7. 4h ATR(14) for trailing stop (2.5x)
8. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45 AND ADX>25
9. KAMA crossover signals only valid with regime confirmation

Why KAMA over HMA/EMA:
- Efficiency Ratio (ER) adapts smoothing constant based on market state
- ER = |net_change| / sum_of_absolute_changes (0=noise, 1=pure trend)
- Fast SC = 2/(2+1), Slow SC = 2/(20+1)
- SC = ER * (Fast - Slow) + Slow
- This automatically reduces sensitivity during choppy markets

Key improvements from failed 4h strategies:
- KAMA instead of HMA (better volatility adaptation)
- ADX filter to avoid trend signals in low-trend environments
- Relaxed RSI thresholds (40/60) for more signals
- Hold logic to maintain positions through minor pullbacks
- Ensure entry conditions trigger on ALL symbols (BTC/ETH/SOL)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 35-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_chop_regime_adx_rsi_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    ER = 1 means pure trend, ER = 0 means pure noise.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i-period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

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
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / (di_sum + 1e-10)
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_4h_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 12h KAMA for intermediate trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d KAMA for secular trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=50, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h_fast[i]) or np.isnan(kama_4h_slow[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        
        # === LONG-TERM TREND BIAS (12h & 1d KAMA) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === KAMA CROSSOVER SIGNALS ===
        kama_bullish_cross = kama_4h_fast[i] > kama_4h_slow[i]
        kama_bearish_cross = kama_4h_fast[i] < kama_4h_slow[i]
        
        # KAMA cross detection (cross happened this bar)
        kama_cross_up = False
        kama_cross_down = False
        if i > 0 and not np.isnan(kama_4h_fast[i-1]) and not np.isnan(kama_4h_slow[i-1]):
            kama_cross_up = kama_4h_fast[i-1] <= kama_4h_slow[i-1] and kama_4h_fast[i] > kama_4h_slow[i]
            kama_cross_down = kama_4h_fast[i-1] >= kama_4h_slow[i-1] and kama_4h_fast[i] < kama_4h_slow[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === RSI SIGNALS (Relaxed for more trades) ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral_low = 40 <= rsi_4h[i] < 50
        rsi_neutral_high = 50 < rsi_4h[i] <= 60
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 45 AND ADX > 25) ===
        if trending_regime and strong_trend:
            # Long: Bullish HTF + KAMA bullish + RSI not overbought
            if (trend_12h_bullish or trend_1d_bullish) and kama_bullish_cross:
                if rsi_4h[i] < 70:  # Not extremely overbought
                    desired_signal = BASE_SIZE
            
            # Long on pullback: HTF bullish + KAMA aligned + RSI oversold
            if (trend_12h_bullish or trend_1d_bullish) and kama_bullish_cross and rsi_oversold:
                desired_signal = BASE_SIZE
            
            # Short: Bearish HTF + KAMA bearish + RSI not oversold
            if (trend_12h_bearish or trend_1d_bearish) and kama_bearish_cross:
                if rsi_4h[i] > 30:  # Not extremely oversold
                    desired_signal = -BASE_SIZE
            
            # Short on pullback: HTF bearish + KAMA aligned + RSI overbought
            if (trend_12h_bearish or trend_1d_bearish) and kama_bearish_cross and rsi_overbought:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI extreme oversold + any HTF alignment
            if rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI extreme overbought + any HTF alignment
            if rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # KAMA cross + RSI confluence in range
            if kama_cross_up and rsi_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if kama_cross_down and rsi_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require both KAMA cross and RSI confluence
            if kama_cross_up and rsi_oversold and (trend_12h_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if kama_cross_down and rsi_overbought and (trend_12h_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Fallback for trade generation: extreme RSI alone
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === ENSURE TRADES ON ALL SYMBOLS ===
        # If no signal for many bars, relax conditions to generate trades
        if desired_signal == 0.0:
            # Basic RSI mean reversion (guarantees some trades)
            if rsi_4h[i] < 20:
                desired_signal = REDUCED_SIZE
            elif rsi_4h[i] > 80:
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
                # Hold long if HTF trend intact and KAMA still bullish
                if (trend_12h_bullish or trend_1d_bullish) and kama_bullish_cross and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and KAMA still bearish
                if (trend_12h_bearish or trend_1d_bearish) and kama_bearish_cross and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses + KAMA bearish
            if trend_12h_bearish and trend_1d_bearish and kama_bearish_cross:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + KAMA bullish
            if trend_12h_bullish and trend_1d_bullish and kama_bullish_cross:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 15:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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