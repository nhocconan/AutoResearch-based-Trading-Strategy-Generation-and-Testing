#!/usr/bin/env python3
"""
Experiment #244: 4h Keltner Channel Breakout + Fisher Transform + ADX Trend Strength
Hypothesis: 4h timeframe needs better entry timing than pure trend following. Keltner Channels
(ADR-based) are more adaptive than Donchian for crypto volatility. Fisher Transform catches
reversals at channel boundaries better than RSI. ADX(14) > 20 filters out choppy periods.
Daily HMA provides trend bias, Weekly HMA confirms macro direction. Volume ratio adds conviction.
This differs from failed 4h strategies by using Fisher for entry timing instead of RSI, and
Keltner instead of Donchian/Supertrend. Position sizing: 0.25 entry, stoploss 2.5*ATR.
Target: Beat Sharpe=0.499 with better entry timing on 4h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_fisher_adx_daily_weekly_hma_volume_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price position within range (0 to 1)
    range_val = highest - lowest
    range_val = np.where(range_val > 0, range_val, 1e-10)
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    plus_di = 100 * (plus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_keltner_channels(high, low, close, atr_period=14, atr_mult=2.0, ema_period=20):
    """Calculate Keltner Channels (EMA +/- ATR multiplier)."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, lower, ema

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner_channels(high, low, close, 14, 2.0, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for breakout/cross detection
    prev_fisher = np.roll(fisher, 1)
    prev_fisher[0] = fisher[0]
    prev_keltner_upper = np.roll(keltner_upper, 1)
    prev_keltner_lower = np.roll(keltner_lower, 1)
    prev_keltner_upper[0] = keltner_upper[0]
    prev_keltner_lower[0] = keltner_lower[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (looser to ensure trades)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # ADX trend strength (must have some trend, but not too strict)
        trend_strength = adx[i] > 20  # Lower threshold for more trades
        
        # DI crossover for direction
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Fisher Transform signals (reversal detection)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_cross_up = prev_fisher[i] < -1.0 and fisher[i] > -1.0
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_cross_down = prev_fisher[i] > 1.0 and fisher[i] < 1.0
        # Extreme oversold/overbought
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # Keltner Channel breakout detection
        breakout_long = close[i] > prev_keltner_upper[i]
        breakout_short = close[i] < prev_keltner_lower[i]
        
        # Price position in channel
        above_mid = close[i] > keltner_mid[i]
        below_mid = close[i] < keltner_mid[i]
        
        # Channel squeeze (low volatility - potential breakout)
        channel_width = (keltner_upper[i] - keltner_lower[i]) / keltner_mid[i]
        prev_width = (prev_keltner_upper[i] - prev_keltner_lower[i]) / keltner_mid[i]
        squeeze = channel_width < prev_width * 0.9  # Width decreasing
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Breakout long with trend and momentum
        if breakout_long:
            if daily_bullish and trend_strength and di_bullish:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and vol_bullish and above_mid:
                new_signal = SIZE_ENTRY
        
        # Fisher reversal from oversold in uptrend
        elif fisher_cross_up:
            if daily_bullish and above_mid:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and di_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # Fisher extreme oversold with trend support
        elif fisher_oversold:
            if daily_bullish and trend_strength:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and above_mid:
                new_signal = SIZE_ENTRY
        
        # Pullback to Keltner mid in uptrend
        elif above_mid and daily_bullish:
            if close[i-1] < keltner_mid[i-1] and close[i] > keltner_mid[i]:
                if di_bullish or vol_bullish:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Breakout short with trend and momentum
        if breakout_short:
            if daily_bearish and trend_strength and di_bearish:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and vol_bearish and below_mid:
                new_signal = -SIZE_ENTRY
        
        # Fisher reversal from overbought in downtrend
        elif fisher_cross_down:
            if daily_bearish and below_mid:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and di_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # Fisher extreme overbought with trend support
        elif fisher_overbought:
            if daily_bearish and trend_strength:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and below_mid:
                new_signal = -SIZE_ENTRY
        
        # Pullback to Keltner mid in downtrend
        elif below_mid and daily_bearish:
            if close[i-1] > keltner_mid[i-1] and close[i] < keltner_mid[i]:
                if di_bearish or vol_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals