#!/usr/bin/env python3
"""
Experiment #454: 4h KAMA Trend with Z-Score Mean Reversion and Volume Flow

Hypothesis: After analyzing 453 failed experiments, Fisher+Choppiness failed 
badly (Sharpe=-10.9 in #442). The key insight is that adaptive indicators 
(KAMA) outperform fixed indicators (EMA/HMA) in crypto's variable volatility.

This strategy uses:
1. KAMA(10,2,30) - Kaufman Adaptive MA adapts to market noise, less whipsaw
2. Z-Score(20) - Statistical mean reversion, more robust than RSI extremes
3. Volume Flow - Taker buy volume ratio confirms institutional interest
4. 1D ADX - Regime filter (trending vs ranging)
5. 1W HMA - Major trend bias (prevents counter-trend disasters)

Entry Logic:
- Long: Price > KAMA + Z-score < -1.5 + Volume flow > 0.55 + 1w HMA bull
- Short: Price < KAMA + Z-score > +1.5 + Volume flow < 0.45 + 1w HMA bear
- Ranging regime (ADX < 20): Looser Z-score thresholds (-1.2 / +1.2)

Exit Logic:
- ATR(14) trailing stop at 2.5x
- Trend reversal (1w HMA flips)
- Z-score mean reversion target (crosses 0)

Position Sizing: 0.28 discrete (conservative for 4h volatility)
Stoploss: 2.5 * ATR(14) trailing

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d, 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_zscore_volume_1d_1w_hma_adaptive_atr_v1"
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
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Calculate KAMA
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_zscore(close, period=20):
    """Calculate Z-score for statistical mean reversion."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std.replace(0, np.inf)
    return zscore.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_volume_flow(taker_buy_volume, volume):
    """Calculate volume flow ratio (taker buy / total volume)."""
    flow = np.zeros(len(volume))
    for i in range(len(volume)):
        if volume[i] > 0:
            flow[i] = taker_buy_volume[i] / volume[i]
        else:
            flow[i] = 0.5
    return flow

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    zscore = calculate_zscore(close, 20)
    vol_flow = calculate_volume_flow(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1D ADX) ===
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] <= 25
        
        # === Z-SCORE THRESHOLDS (Adaptive to regime) ===
        if trending_regime:
            zscore_long_threshold = -1.5
            zscore_short_threshold = 1.5
        else:
            zscore_long_threshold = -1.2
            zscore_short_threshold = 1.2
        
        # === VOLUME FLOW CONFIRMATION ===
        vol_flow_bull = vol_flow[i] > 0.55
        vol_flow_bear = vol_flow[i] < 0.45
        
        # === PRICE vs KAMA ===
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Z-score oversold + volume confirmation + trend bias
        if zscore[i] < zscore_long_threshold and vol_flow_bull and price_above_kama:
            if bull_trend_1w:
                new_signal = SIZE
            elif ranging_regime:
                # Allow long in ranging market even without 1w bull trend
                new_signal = SIZE
        
        # SHORT ENTRY: Z-score overbought + volume confirmation + trend bias
        if zscore[i] > zscore_short_threshold and vol_flow_bear and price_below_kama:
            if bear_trend_1w:
                new_signal = -SIZE
            elif ranging_regime:
                # Allow short in ranging market even without 1w bear trend
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === Z-SCORE MEAN REVERSION EXIT ===
        # Exit long when z-score crosses above 0 (mean reached)
        # Exit short when z-score crosses below 0 (mean reached)
        if in_position and new_signal != 0.0:
            if position_side > 0 and i > 0 and not np.isnan(zscore[i-1]):
                if zscore[i-1] < 0 and zscore[i] >= 0:
                    new_signal = 0.0  # Take profit at mean
            if position_side < 0 and i > 0 and not np.isnan(zscore[i-1]):
                if zscore[i-1] > 0 and zscore[i] <= 0:
                    new_signal = 0.0  # Take profit at mean
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w and not ranging_regime:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w and not ranging_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals