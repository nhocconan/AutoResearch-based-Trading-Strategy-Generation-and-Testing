#!/usr/bin/env python3
"""
Experiment #844: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 580+ failed strategies, the key insight is that 4h timeframe with
adaptive trend filtering (KAMA) works better than static EMAs in bear/range markets.
KAMA adapts to volatility - fast in trends, slow in chop. Combined with ADX for
trend strength and Choppiness for regime detection, this should outperform static MA strategies.

Strategy design:
1. 4h Primary timeframe (target 30-50 trades/year)
2. 12h KAMA(10) for adaptive trend bias (not entry trigger)
3. 1d HMA(21) for secular trend filter (long-term direction)
4. 4h ADX(14) for trend strength confirmation (>25 = trending)
5. 4h Choppiness Index(14) for regime detection (>55 = range, <45 = trend)
6. 4h RSI(14) with relaxed thresholds (35/65) for entry timing
7. 4h ATR(14) for trailing stop (2.5x)
8. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45
9. Volume confirmation: taker_buy_volume ratio > 0.55 for longs

Why KAMA:
- Kaufman Adaptive Moving Average adjusts smoothing based on market noise
- Efficiency Ratio (ER) = |close - close_n| / sum(|close - close_prev|)
- Fast SC = 2/(2+1), Slow SC = 2/(20+1)
- SC = ER * (FastSC - SlowSC) + SlowSC
- KAMA = KAMA_prev + SC * (close - KAMA_prev)
- Works well in 2022 crash and 2025 bear market (adapts to vol spikes)

Key changes from failed strategies:
- KAMA instead of HMA/EMA for primary trend (adaptive to volatility)
- ADX threshold: 25 (not 30) — more signals while filtering noise
- CHOP thresholds: 55/45 (clearer regime separation)
- RSI thresholds: 35/65 (not 30/70) — more signals on 4h
- Volume confirmation for entry quality
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_regime_rsi_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10):
    """
    Kaufman Adaptive Moving Average — adjusts smoothing based on market noise.
    Fast SC = 2/(2+1), Slow SC = 2/(20+1)
    ER = |close - close_n| / sum(|close - close_prev|) over period
    SC = ER * (FastSC - SlowSC) + SlowSC
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (20.0 + 1.0)
    
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        # Efficiency Ratio
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if noise < 1e-10:
            er = 1.0
        else:
            er = signal / noise
        
        # Smoothing Constant
        sc = er * (fast_sc - slow_sc) + slow_sc
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
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
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
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
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 12h KAMA for medium-term trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # Volume ratio for confirmation
        if volume[i] > 1e-10:
            volume_ratio = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio = 0.5
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND BIAS (12h HTF KAMA10) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA10) ===
        trend_4h_bullish = close[i] > kama_4h[i]
        trend_4h_bearish = close[i] < kama_4h[i]
        kama_slope_up = kama_4h[i] > kama_4h[i-1] if not np.isnan(kama_4h[i-1]) else False
        kama_slope_down = kama_4h[i] < kama_4h[i-1] if not np.isnan(kama_4h[i-1]) else False
        
        # === TREND STRENGTH (4h ADX14) ===
        trending_market = adx_4h[i] > 25
        ranging_market = adx_4h[i] < 20
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS (Relaxed for 4h timeframe) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral_low = 35 <= rsi_4h[i] < 50
        rsi_neutral_high = 50 < rsi_4h[i] <= 65
        rsi_cross_up = rsi_4h[i-1] < 35 and rsi_4h[i] >= 35 if not np.isnan(rsi_4h[i-1]) else False
        rsi_cross_down = rsi_4h[i-1] > 65 and rsi_4h[i] <= 65 if not np.isnan(rsi_4h[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = volume_ratio > 0.55
        volume_bearish = volume_ratio < 0.45
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55 or ADX < 20) — Mean Reversion ===
        if ranging_regime or ranging_market:
            # Long: RSI oversold + price below KAMA + volume support
            if rsi_oversold and trend_4h_bearish and volume_bullish:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + price above KAMA + volume support
            if rsi_overbought and trend_4h_bullish and volume_bearish:
                desired_signal = -BASE_SIZE
            
            # RSI cross + trend alignment (high probability)
            if rsi_cross_up and rsi_4h[i] < 45 and (trend_12h_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_cross_down and rsi_4h[i] > 55 and (trend_12h_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Fallback: extreme RSI alone (guarantees trades on all symbols)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45 and ADX > 25) — Trend Following ===
        elif trending_regime and trending_market:
            # Long: All timeframes bullish + RSI pullback + volume
            if trend_1d_bullish and trend_12h_bullish and trend_4h_bullish:
                if rsi_neutral_low and volume_bullish:
                    desired_signal = BASE_SIZE
                elif kama_slope_up and rsi_4h[i] < 60:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: All timeframes bearish + RSI pullback + volume
            if trend_1d_bearish and trend_12h_bearish and trend_4h_bearish:
                if rsi_neutral_high and volume_bearish:
                    desired_signal = -BASE_SIZE
                elif kama_slope_down and rsi_4h[i] > 40:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Breakout: KAMA cross with ADX confirmation
            if kama_slope_up and adx_4h[i] > 30 and rsi_4h[i] < 70:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if kama_slope_down and adx_4h[i] > 30 and rsi_4h[i] > 30:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55 or 20 <= ADX <= 25) ===
        else:
            # Conservative: RSI + KAMA confluence + HTF alignment
            if rsi_oversold and trend_4h_bullish and (trend_12h_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and trend_4h_bearish and (trend_12h_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Basic mean reversion with single HTF filter
            if rsi_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_overbought and trend_1d_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if (trend_12h_bullish or trend_1d_bullish) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if (trend_12h_bearish or trend_1d_bearish) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if all timeframes reverse + RSI overbought
            if trend_1d_bearish and trend_12h_bearish and rsi_4h[i] > 75:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if all timeframes reverse + RSI oversold
            if trend_1d_bullish and trend_12h_bullish and rsi_4h[i] < 25:
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