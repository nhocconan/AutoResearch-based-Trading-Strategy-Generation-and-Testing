#!/usr/bin/env python3
"""
Experiment #180: 1d Adaptive Trend-Mean Reversion with Weekly HMA Filter
Hypothesis: Daily timeframe captures major swings while Weekly HMA provides 
macro trend bias. Strategy adapts between trend-following (CHOP<45) and 
mean-reversion (CHOP>55) modes. Entry conditions loosened for sufficient 
trades on daily data (RSI 35-65 range, not extremes). ATR stoploss at 2.5*ATR.
Volume confirmation ensures breakout validity. Position sizing: 0.25 entry, 
0.125 half-size at 2R profit. Discrete levels minimize fee churn.
This targets 2022 crash (trend mode shorts) and 2025 consolidation (range mode).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_adaptive_trend_mr_weekly_hma_chop_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = pd.Series(close_s).diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = np.where(volatility > 0, change / volatility, 0.0)
    er = np.nan_to_num(er, 0.0)
    
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    return vol_ratio

def calculate_momentum(close, period=10):
    """Calculate rate of change momentum."""
    close_s = pd.Series(close)
    mom = close_s.pct_change(periods=period).values
    mom = np.where(np.isnan(mom), 0.0, mom)
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    kama = calculate_kama(close, 10, 2, 30)
    vol_ratio = calculate_volume_ratio(volume, 20)
    momentum = calculate_momentum(close, 10)
    
    # Calculate BB for mean reversion
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_std = np.where(bb_std > 0, bb_std, 1e-10)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
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
        # HTF trend filters (weekly bias)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Regime detection (loosened thresholds for more trades)
        is_ranging = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # 1d trend
        trend_bullish = hma_21[i] > hma_50[i] and close[i] > hma_21[i]
        trend_bearish = hma_21[i] < hma_50[i] and close[i] < hma_21[i]
        
        # KAMA trend confirmation
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # RSI signals (loosened for more trades on daily)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        rsi_neutral = 35 < rsi[i] < 65
        
        # Momentum
        mom_positive = momentum[i] > 0.02
        mom_negative = momentum[i] < -0.02
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.0
        
        # BB position
        near_lower = close[i] < bb_lower[i] * 1.01
        near_upper = close[i] > bb_upper[i] * 0.99
        near_mid = bb_lower[i] < close[i] < bb_upper[i]
        
        new_signal = 0.0
        
        # === TREND FOLLOWING MODE ===
        if is_trending:
            # Long: trend bullish + RSI pullback + weekly not bearish
            if trend_bullish and kama_bullish:
                if rsi_neutral and rsi_rising and not weekly_bearish:
                    if mom_positive or volume_confirmed:
                        new_signal = SIZE_ENTRY
            
            # Short: trend bearish + RSI pullback + weekly not bullish
            elif trend_bearish and kama_bearish:
                if rsi_neutral and rsi_falling and not weekly_bullish:
                    if mom_negative or volume_confirmed:
                        new_signal = -SIZE_ENTRY
            
            # Breakout continuation
            elif trend_bullish and close[i] > hma_21[i] * 1.01:
                if weekly_bullish and volume_confirmed:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and close[i] < hma_21[i] * 0.99:
                if weekly_bearish and volume_confirmed:
                    new_signal = -SIZE_ENTRY
        
        # === MEAN REVERSION MODE ===
        elif is_ranging:
            # Long: near lower BB + RSI oversold + weekly not strongly bearish
            if near_lower and rsi_oversold:
                if not weekly_bearish or rsi_rising:
                    new_signal = SIZE_ENTRY
            
            # Short: near upper BB + RSI overbought + weekly not strongly bullish
            elif near_upper and rsi_overbought:
                if not weekly_bullish or rsi_falling:
                    new_signal = -SIZE_ENTRY
            
            # Mid-range mean reversion
            elif near_mid and rsi[i] < 40 and rsi_rising:
                if not weekly_bearish:
                    new_signal = SIZE_ENTRY
            elif near_mid and rsi[i] > 60 and rsi_falling:
                if not weekly_bullish:
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