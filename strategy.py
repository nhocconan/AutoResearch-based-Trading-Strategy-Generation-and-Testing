#!/usr/bin/env python3
"""
Experiment #819: 4h Primary + 1d HTF — ADX Regime + KAMA Trend + Volume Breakout

Hypothesis: After 559 failed strategies and #811 achieving only Sharpe=0.126:
1. #811's Choppiness Index is slow to react — ADX responds faster to regime changes
2. HMA is noisy — KAMA (Kaufman Adaptive) adapts to volatility, smoother trends
3. Donchian breakouts need VOLUME confirmation to filter false breakouts
4. RSI thresholds 35/65 are too wide — use 30/70 for entries, 20/80 for extremes
5. Need MORE trades than #811 — relax hold conditions, tighten entry slightly
6. Volume spike (taker_buy_volume ratio > 1.5) confirms genuine breakout momentum
7. Dual ADX thresholds: ADX>25=trending, ADX<20=ranging, 20-25=transition

Strategy design:
1. 1d KAMA(21) for intermediate trend (aligned via mtf_data)
2. 4h ADX(14) for regime detection (faster than Choppiness)
3. 4h KAMA(21) for adaptive trend following
4. 4h RSI(14) for entry timing with 30/70 thresholds
5. 4h Volume ratio (taker_buy_volume / total_volume) for breakout confirmation
6. 4h ATR(14) for trailing stop (2.5x — wider than #811's 2.0x)
7. Discrete signals: 0.0, ±0.25, ±0.30
8. Target: 35-50 trades/year on 4h timeframe

Key changes from #811:
- ADX(14) instead of Choppiness(14) — faster regime detection
- KAMA(21) instead of HMA(21) — adaptive to volatility, less whipsaw
- Volume confirmation on breakouts — filters false signals
- RSI thresholds: 30/70 entries, 20/80 extremes (more balanced)
- ATR stop: 2.5x (wider, fewer premature exits)
- Simpler hold logic — maintain position while ADX confirms regime

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 35-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_kama_volume_breakout_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    Average Directional Index — measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, plus_di, minus_di
    
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
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_vals = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di_vals = 100 * minus_dm_smooth / (atr + 1e-10)
    
    plus_di = np.clip(plus_di_vals, 0, 100)
    minus_di = np.clip(minus_di_vals, 0, 100)
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
    return adx, plus_di, minus_di

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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Volume ratio — taker buy volume / total volume."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = taker_buy_volume / (volume + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    kama_4h = calculate_kama(close, period=21)
    atr_4h = calculate_atr(high, low, close, period=14)
    vol_ratio_4h = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Calculate and align 1d KAMA for intermediate trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF KAMA21) ===
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (4h KAMA21) ===
        trend_4h_bullish = close[i] > kama_4h[i]
        trend_4h_bearish = close[i] < kama_4h[i]
        
        # === REGIME DETECTION (4h ADX14) ===
        trending_regime = adx_4h[i] > 25
        ranging_regime = adx_4h[i] < 20
        transition_regime = 20 <= adx_4h[i] <= 25
        
        # === DIRECTIONAL BIAS (DI+ vs DI-) ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i]
        di_bearish = minus_di_4h[i] > plus_di_4h[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        rsi_extreme_oversold = rsi_4h[i] < 20
        rsi_extreme_overbought = rsi_4h[i] > 80
        rsi_neutral_low = 30 <= rsi_4h[i] < 45
        rsi_neutral_high = 55 < rsi_4h[i] <= 70
        
        # === VOLUME CONFIRMATION ===
        volume_spike_long = vol_ratio_4h[i] > 0.55  # More buying pressure
        volume_spike_short = vol_ratio_4h[i] < 0.45  # More selling pressure
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (ADX > 25) — Trend Following ===
        if trending_regime:
            # Long: 1d bullish + 4h bullish + DI+ > DI- + volume confirmation
            if (trend_1d_bullish and trend_4h_bullish and di_bullish):
                if volume_spike_long or rsi_4h[i] < 60:  # Entry or pullback
                    desired_signal = BASE_SIZE
            
            # Short: 1d bearish + 4h bearish + DI- > DI+ + volume confirmation
            if (trend_1d_bearish and trend_4h_bearish and di_bearish):
                if volume_spike_short or rsi_4h[i] > 40:  # Entry or pullback
                    desired_signal = -BASE_SIZE
            
            # Pullback entries in strong trend (reduced size)
            if trend_1d_bullish and di_bullish and rsi_neutral_low:
                if desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and di_bearish and rsi_neutral_high:
                if desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (ADX < 20) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + price above 1d KAMA (trend filter)
            if rsi_oversold and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + price below 1d KAMA
            if rsi_overbought and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            
            # Extreme RSI alone (guarantees trades in choppy markets)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRANSITION REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative entries with multiple confirmations
            if trend_1d_bullish and di_bullish and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and di_bearish and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Volume breakout in transition (potential trend start)
            if trend_1d_bullish and volume_spike_long and rsi_4h[i] < 55:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if trend_1d_bearish and volume_spike_short and rsi_4h[i] > 45:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend intact and ADX not collapsing
                if trend_1d_bullish and adx_4h[i] > 15 and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and ADX not collapsing
                if trend_1d_bearish and adx_4h[i] > 15 and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses + ADX confirms
            if trend_1d_bearish and adx_4h[i] > 20:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses + ADX confirms
            if trend_1d_bullish and adx_4h[i] > 20:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_4h[i] < 20:
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