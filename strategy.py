#!/usr/bin/env python3
"""
Experiment #006: 12h Bollinger Squeeze Breakout with 1d Trend Filter

Hypothesis: Previous regime-switching strategies failed because they over-complicated
entry logic. This strategy uses a SIMPLE but proven pattern:
1. Bollinger Band Squeeze (low volatility compression)
2. Breakout from squeeze with volume confirmation
3. 1d HMA for major trend bias (only trade with HTF trend)
4. ATR stoploss for risk management

Why this should work:
1. BB Squeeze is a proven volatility breakout pattern (John Carter, TTM Squeeze)
2. Works in BOTH bull and bear markets (captures explosive moves either direction)
3. 12h timeframe naturally filters noise (20-50 trades/year target)
4. 1d HMA provides major trend alignment (don't fight the higher TF)
5. Volume confirmation reduces false breakouts

Key differences from failed experiments:
- NO Connors RSI (failed 3 times already)
- NO Choppiness Index regime switching (failed 3 times already)
- SIMPLE breakout logic with clear entry/exit
- Volume filter to confirm breakout validity

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_squeeze_breakout_1d_trend_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (normalized)."""
    sma = np.where(sma == 0, 1e-10, sma)
    bb_width = (upper - lower) / sma
    return bb_width

def calculate_bb_percentile(close, upper, lower, period=100):
    """Calculate where price sits within BB (0-100 scale)."""
    bb_range = upper - lower
    bb_range = np.where(bb_range == 0, 1e-10, bb_range)
    bb_pct = (close - lower) / bb_range * 100
    return bb_pct

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - moves fast in trends, slow in ranges.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1))
    noise_sum = noise.rolling(window=er_period, min_periods=er_period).sum()
    noise_sum = noise_sum.replace(0, 1e-10)
    er = signal / noise_sum
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_pct = calculate_bb_percentile(close, bb_upper, bb_lower)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    kama_12h = calculate_kama(close, 10, 2, 30)
    
    # Calculate BB Width percentile (for squeeze detection)
    bb_width_sma = pd.Series(bb_width).rolling(window=100, min_periods=100).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=100, min_periods=100).std().values
    bb_width_zscore = (bb_width - bb_width_sma) / np.where(bb_width_std == 0, 1e-10, bb_width_std)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_zscore[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        daily_neutral = not daily_bullish and not daily_bearish
        
        # === BB SQUEEZE DETECTION ===
        # Squeeze = BB Width at 6-month low (z-score < -1.5)
        is_squeeze = bb_width_zscore[i] < -1.0
        
        # === VOLUME CONFIRMATION ===
        current_vol = volume[i]
        vol_ratio = current_vol / vol_ma_20[i]
        high_volume = vol_ratio > 1.3  # 30% above average
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (BB SQUEEZE BREAKOUT) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: Squeeze + Bullish breakout + Volume + Daily trend support
        if is_squeeze and kama_bullish:
            # Check for breakout above BB upper
            if close[i] > bb_upper[i] and high_volume:
                if daily_bullish or daily_neutral:
                    new_signal = current_size
            
            # Alternative: Break above recent high with volume
            elif close[i] > pd.Series(high).iloc[max(0,i-20):i+1].max() and high_volume:
                if daily_bullish:
                    new_signal = current_size * 0.8
        
        # SHORT: Squeeze + Bearish breakout + Volume + Daily trend support
        elif is_squeeze and kama_bearish:
            # Check for breakdown below BB lower
            if close[i] < bb_lower[i] and high_volume:
                if daily_bearish or daily_neutral:
                    new_signal = -current_size
            
            # Alternative: Break below recent low with volume
            elif close[i] < pd.Series(low).iloc[max(0,i-20):i+1].min() and high_volume:
                if daily_bearish:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), force entry with weaker conditions
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and vol_ratio > 1.1:
                new_signal = current_size * 0.5
            elif kama_bearish and daily_bearish and vol_ratio > 1.1:
                new_signal = -current_size * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and daily_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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