#!/usr/bin/env python3
"""
Experiment #322: 4h KAMA Trend + MACD Momentum + Dual HTF HMA Regime Filter

Hypothesis: #316 showed regime detection (CHOP) works on 4h with Sharpe=0.676.
However, many strategies failed with Supertrend (#310, #319) and RSI (#312, #313).

This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than EMA
2. MACD histogram for momentum confirmation (proven edge in trend markets)
3. 1d HMA(21) for primary directional bias
4. 1w HMA(21) for meta-trend confirmation
5. Choppiness Index for regime detection (range vs trend)
6. ATR(14) trailing stoploss at 2.0x (tighter than #316's 2.5x)

Key differences from #316:
- KAMA instead of HMA for primary trend signal (more adaptive)
- MACD histogram momentum filter (not just CHOP regime)
- Dual HTF bias (both 1d AND 1w required for full position)
- Tighter stoploss (2.0 ATR vs 2.5 ATR)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_macd_dual_htf_chop_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trending markets, slow in ranging.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    er = change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop[0:period] = np.nan
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
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(kama[i]) or np.isnan(macd_hist[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias (REQUIRED)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation (REQUIRED for full size)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP < 38.2 = trending market (favor trend following)
        # CHOP > 61.8 = ranging market (favor mean reversion)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # === KAMA TREND ===
        # Price above KAMA = bullish trend
        # Price below KAMA = bearish trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (momentum)
        kama_slope_bull = kama[i] > kama[i-5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i-5] if i >= 5 else False
        
        # === MACD MOMENTUM ===
        # MACD histogram above 0 = bullish momentum
        # MACD histogram below 0 = bearish momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # MACD histogram increasing = strengthening momentum
        macd_hist_increasing = macd_hist[i] > macd_hist[i-1] if i >= 1 else False
        macd_hist_decreasing = macd_hist[i] < macd_hist[i-1] if i >= 1 else False
        
        # === DETERMINE POSITION SIZE ===
        # Full size when both HTF agree + trending regime
        if bull_trend_1d and bull_trend_1w and trending_regime:
            position_size = SIZE_STRONG
        elif bear_trend_1d and bear_trend_1w and trending_regime:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1d bias up + 1w bias up + KAMA bullish + MACD bullish + trending regime
        # Relaxed: only need 1d bias + KAMA + MACD (1w boosts size)
        long_conditions = (
            bull_trend_1d and
            kama_bullish and
            macd_bullish and
            (trending_regime or kama_slope_bull)
        )
        
        # SHORT: 1d bias down + 1w bias down + KAMA bearish + MACD bearish + trending regime
        short_conditions = (
            bear_trend_1d and
            kama_bearish and
            macd_bearish and
            (trending_regime or kama_slope_bear)
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
                new_signal = 0.0
        
        # === MACD REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bearish and macd_hist_decreasing:
                new_signal = 0.0
            if position_side < 0 and macd_bullish and macd_hist_increasing:
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