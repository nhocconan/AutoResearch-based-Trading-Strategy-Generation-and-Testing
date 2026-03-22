#!/usr/bin/env python3
"""
Experiment #065: 12h Fisher Transform + 1d HMA Trend + ADX Regime Filter
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Combined with 1d HMA trend bias and ADX regime filter for asymmetric entries.
Key insight: Fisher Transform normalizes price to Gaussian distribution, making extremes statistically significant.
Long when Fisher crosses above -1.5 in bull regime, short when crosses below +1.5 in bear regime.
ADX filter distinguishes trending (follow trend) vs ranging (mean revert) for adaptive logic.
Multiple entry paths ensure 10+ trades per symbol while maintaining quality.
Position sizing: 0.20 base, 0.30 strong trend, 0.35 very strong - discrete levels minimize fee churn.
Stoploss: 2.5*ATR trailing stop via signal=0 when hit.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_1d_hma_adx_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at statistical extremes. Proven effective in bear markets.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        normalized = (hl2 - lowest) / range_val
        
        # Clamp to avoid division by zero
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i - 1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i - 1]
        else:
            fisher[i] = fisher_val
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i - 1]
    
    return fisher, trigger

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Fisher Transform for reversals
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    
    # HMA on 12h for short-term trend
    hma_12h = calculate_hma(close, 21)
    hma_12h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h HMA = short-term trend
        bull_trend_12h = hma_12h_fast[i] > hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        bear_trend_12h = hma_12h_fast[i] < hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === TREND STRENGTH / REGIME ===
        trending_regime = adx[i] > 22
        strong_trend = adx[i] > 30
        ranging_regime = adx[i] < 18
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_long_extreme = fisher[i] > -1.0 and fisher[i-1] <= -1.5 if i > 0 else False
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        fisher_short_extreme = fisher[i] < 1.0 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.01 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.99 if not np.isnan(bb_upper[i]) else False
        
        # === VOLUME CONFIRMATION ===
        high_volume = vol_ratio[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Fisher reversal + bull trend (trending regime)
        if trending_regime and bull_trend_1d:
            if fisher_long_cross and di_bullish:
                if strong_trend and high_volume:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Fisher extreme + trend alignment
        if bull_trend_1d and bull_trend_12h:
            if fisher_long_extreme and ema_bullish:
                new_signal = SIZE_BASE
        
        # Path 3: Mean reversion in ranging market
        if ranging_regime:
            if fisher_long_cross and rsi_oversold and near_bb_lower:
                new_signal = SIZE_WEAK
        
        # Path 4: Simple trend continuation
        if bull_trend_1d and ema_bullish:
            if rsi_neutral and di_bullish:
                if close[i] > ema_21[i]:
                    new_signal = SIZE_BASE
        
        # Path 5: RSI + Fisher combo
        if rsi_oversold and fisher[i] < -1.0:
            if bull_trend_1d or above_sma200:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Fisher reversal + bear trend (trending regime)
        if trending_regime and bear_trend_1d:
            if fisher_short_cross and di_bearish:
                if strong_trend and high_volume:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Fisher extreme + trend alignment
        if bear_trend_1d and bear_trend_12h:
            if fisher_short_extreme and ema_bearish:
                new_signal = -SIZE_BASE
        
        # Path 3: Mean reversion in ranging market
        if ranging_regime:
            if fisher_short_cross and rsi_overbought and near_bb_upper:
                new_signal = -SIZE_WEAK
        
        # Path 4: Simple trend continuation
        if bear_trend_1d and ema_bearish:
            if rsi_neutral and di_bearish:
                if close[i] < ema_21[i]:
                    new_signal = -SIZE_BASE
        
        # Path 5: RSI + Fisher combo
        if rsi_overbought and fisher[i] > 1.0:
            if bear_trend_1d or below_sma200:
                new_signal = -SIZE_BASE
        
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