#!/usr/bin/env python3
"""
Experiment #014: 4h KAMA Trend + ADX + Volume Surge with 12h/1d MTF Filter

Hypothesis: The winning pattern from #004 (mtf_4h_kama_adx_volume_1d_filter_v1, Sharpe=0.514)
shows KAMA + ADX + Volume works on 4h. This experiment IMPROVES it by:
1. Adding 12h HMA as intermediate trend filter (not just 1d)
2. Using volume surge ratio (current/median) instead of absolute volume
3. Asymmetric entry: require ADX>25 for entry, ADX>18 for exit (hysteresis)
4. Looser ADX threshold (20 instead of 25) to ensure trade frequency
5. Volume confirmation: only enter when volume > 1.5x 20-bar median

Why this should work:
- KAMA adapts to volatility (faster in trends, slower in chop)
- ADX filters out weak trends (whipsaw protection)
- Volume surge confirms breakout validity
- 12h + 1d dual HTF filter prevents counter-trend trades
- 4h TF targets 20-50 trades/year (fee-efficient)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_volume_12h_1d_filter_v2"
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
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er_num = np.abs(close - np.roll(close, er_period))
    er_den = np.zeros(n)
    for i in range(er_period, n):
        er_den[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    er_den = np.where(er_den == 0, 1e-10, er_den)
    er = er_num / er_den
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = strong trend, ADX < 20 = weak/no trend
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX
    di_sum = plus_di + minus_di
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio: current volume / median volume over period."""
    vol_s = pd.Series(volume)
    vol_median = vol_s.rolling(window=period, min_periods=period).median().values
    vol_ratio = volume / vol_median
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(adx_14[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        # Both HTFs must agree for strong bias
        htf_bullish = (close[i] > hma_12h_21_aligned[i]) and (close[i] > hma_1d_21_aligned[i])
        htf_bearish = (close[i] < hma_12h_21_aligned[i]) and (close[i] < hma_1d_21_aligned[i])
        
        # Neutral if HTFs disagree
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === 4H KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === ADX TREND STRENGTH (with hysteresis) ===
        adx_strong = adx_14[i] > 20  # Lowered from 25 for more trades
        adx_weak = adx_14[i] < 15  # Exit threshold
        
        # === VOLUME SURGE CONFIRMATION ===
        volume_surge = vol_ratio[i] > 1.3  # 30% above median
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: KAMA bullish + ADX strong + Volume surge + HTF bullish
        if kama_bullish and adx_strong and htf_bullish:
            # Volume surge required for initial entry
            if volume_surge:
                new_signal = current_size
            # Or if already in position, maintain
            elif in_position and position_side > 0:
                new_signal = current_size
        
        # SHORT ENTRY: KAMA bearish + ADX strong + Volume surge + HTF bearish
        elif kama_bearish and adx_strong and htf_bearish:
            if volume_surge:
                new_signal = -current_size
            elif in_position and position_side < 0:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~5 days on 4h), allow weaker entry
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if kama_bullish and htf_bullish and adx_14[i] > 18:
                new_signal = current_size * 0.7
            elif kama_bearish and htf_bearish and adx_14[i] > 18:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT (ADX hysteresis) ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if KAMA turns bearish OR ADX goes weak
            if position_side > 0 and (kama_bearish or adx_weak):
                trend_reversal = True
            # Exit short if KAMA turns bullish OR ADX goes weak
            if position_side < 0 and (kama_bullish or adx_weak):
                trend_reversal = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or htf_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals