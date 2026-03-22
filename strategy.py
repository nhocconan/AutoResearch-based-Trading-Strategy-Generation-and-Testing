#!/usr/bin/env python3
"""
Experiment #017: 12h Donchian Breakout with 1d HMA Regime + Volume Confirmation
Hypothesis: 12h Donchian breakouts capture sustained moves while 1d HMA filters counter-trend breakouts.
Key insight: Previous 12h strategies failed due to too many conflicting filters or wrong indicator combo.
This uses cleaner logic: Donchian(20) breakout + 1d HMA trend bias + volume spike confirmation + ATR stops.
Why 12h Donchian: Fewer false breakouts than 4h, catches multi-day trends, works in both bull/bear regimes.
Volume filter: Breakout volume > 1.5x 20-bar avg = real move, not fake breakout.
Position sizing: 0.25 base, 0.35 on high-conviction (volume > 2x), discrete levels to minimize churn.
Must generate 10+ trades on train - Donchian(20) on 12h = ~2-3 breakouts/month = 24-36/year.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_vol_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_fisher_transform(high, low, close, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    tp = (high + low + close) / 2
    
    # Normalize to -1 to +1 range
    highest = np.zeros(n)
    lowest = np.zeros(n)
    
    for i in range(period - 1, n):
        highest[i] = np.max(tp[i - period + 1:i + 1])
        lowest[i] = np.min(tp[i - period + 1:i + 1])
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 2 * (tp - lowest) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (EMA of Fisher)
    fisher = pd.Series(fisher_raw).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Donchian Channel (20-period breakout)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Fisher Transform for reversal confirmation
    fisher = calculate_fisher_transform(high, low, close, 9)
    
    # Bollinger Bands for squeeze detection
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track last breakout direction to avoid whipsaws
    last_breakout_dir = 0  # 0=none, 1=long, -1=short
    last_breakout_bar = -100
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h trend confirmation
        bull_trend_12h = close[i] > ema_50[i]
        bear_trend_12h = close[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Volume confirmation
        vol_ratio = volume[i] / (vol_sma[i] + 1e-10) if vol_sma[i] > 0 else 1.0
        high_volume = vol_ratio > 1.5
        very_high_volume = vol_ratio > 2.0
        
        # Donchian breakout detection
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Fisher Transform signals (reversal confirmation)
        fisher_oversold = not np.isnan(fisher[i]) and fisher[i] < -1.5
        fisher_overbought = not np.isnan(fisher[i]) and fisher[i] > 1.5
        
        # RSI confirmation (not too extended)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # BB squeeze (low volatility before breakout)
        bb_squeeze = not np.isnan(bb_width[i]) and bb_width[i] < np.nanpercentile(bb_width[:i], 30) if i > 100 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Only enter long when 1d trend is bullish OR strong breakout with volume
        if bull_trend_1d or (breakout_long and very_high_volume):
            # Primary: Donchian breakout with volume confirmation
            if breakout_long and high_volume and rsi_not_overbought:
                # Check we didn't just breakout in opposite direction (whipsaw filter)
                if last_breakout_dir != -1 or (i - last_breakout_bar) > 10:
                    size = SIZE_HIGH if very_high_volume else SIZE_BASE
                    new_signal = size
                    last_breakout_dir = 1
                    last_breakout_bar = i
            
            # Secondary: Fisher oversold reversal in uptrend
            elif fisher_oversold and bull_trend_1d and rsi[i] < 45:
                new_signal = SIZE_HALF
            
            # Tertiary: Pullback to EMA50 with volume support
            elif close[i] <= ema_50[i] * 1.02 and close[i] >= ema_50[i] * 0.98 and vol_ratio > 1.2 and bull_trend_12h:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        # Only enter short when 1d trend is bearish OR strong breakout with volume
        elif bear_trend_1d or (breakout_short and very_high_volume):
            # Primary: Donchian breakdown with volume confirmation
            if breakout_short and high_volume and rsi_not_oversold:
                # Check we didn't just breakout in opposite direction (whipsaw filter)
                if last_breakout_dir != 1 or (i - last_breakout_bar) > 10:
                    size = SIZE_HIGH if very_high_volume else SIZE_BASE
                    new_signal = -size
                    last_breakout_dir = -1
                    last_breakout_bar = i
            
            # Secondary: Fisher overbought reversal in downtrend
            elif fisher_overbought and bear_trend_1d and rsi[i] > 55:
                new_signal = -SIZE_HALF
            
            # Tertiary: Bounce to EMA50 with volume support
            elif close[i] >= ema_50[i] * 0.98 and close[i] <= ema_50[i] * 1.02 and vol_ratio > 1.2 and bear_trend_12h:
                new_signal = -SIZE_HALF
        
        # === CONTRARIAN ENTRIES (when extreme moves) ===
        # Long when price < BB lower + RSI < 25 (oversold bounce)
        if rsi[i] < 25 and close[i] < bb_lower[i] and not np.isnan(bb_lower[i]):
            if position_side <= 0:  # Only if not already long
                new_signal = SIZE_HALF
        
        # Short when price > BB upper + RSI > 75 (overbought rejection)
        if rsi[i] > 75 and close[i] > bb_upper[i] and not np.isnan(bb_upper[i]):
            if position_side >= 0:  # Only if not already short
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals