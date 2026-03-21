#!/usr/bin/env python3
"""
Experiment #441: 1h Regime-Adaptive Strategy with 4h HMA Bias
Hypothesis: Different market regimes require different approaches. 
- Trending regimes (CHOP < 38.2): Follow 4h HMA trend with 1h momentum
- Ranging regimes (CHOP > 61.8): Mean revert with RSI extremes + Bollinger
- Transition regimes: Stay flat or reduce position size
This adapts to market conditions instead of using one rigid approach.
4h HMA provides higher timeframe bias (proven in successful strategies).
Position size: 0.25 discrete, stoploss 2.5*ATR, ensure >=10 trades per symbol.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_chop_rsi_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            atr_sum += max(tr1, tr2, tr3)
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, 20, 2.0)
    ema21 = calculate_ema(close, 21)
    ema50 = calculate_ema(close, 50)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        in_trend_regime = chop[i] < 45.0  # Trending market
        in_range_regime = chop[i] > 55.0  # Ranging market
        # Neutral regime: 45-55, reduce position or stay flat
        
        # === 4H TREND BIAS ===
        hma_bullish = close[i] > hma_4h_aligned[i]
        hma_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1H TREND FILTERS ===
        ema_bullish = ema21[i] > ema50[i]
        ema_bearish = ema21[i] < ema50[i]
        price_above_ema21 = close[i] > ema21[i]
        price_below_ema21 = close[i] < ema21[i]
        
        # === MOMENTUM FILTERS ===
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if not np.isnan(macd_hist[i-1]) else macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if not np.isnan(macd_hist[i-1]) else macd_hist[i] < 0
        adx_strong = adx[i] > 20
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === RSI FILTERS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 65
        rsi_neutral_short = rsi[i] > 35 and rsi[i] < 60
        
        # === BOLLINGER FILTERS ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_sma[i] < 0.10
        
        new_signal = 0.0
        
        # === TREND REGIME ENTRIES (CHOP < 45) ===
        if in_trend_regime:
            # Long: 4h HMA bullish + 1h EMA bullish + MACD positive + ADX strong
            if hma_bullish and ema_bullish and macd_bullish and adx_strong and rsi_neutral_long:
                new_signal = SIZE_ENTRY
            # Long: 4h HMA bullish + Price above EMA21 + DI bullish + RSI > 45
            elif hma_bullish and price_above_ema21 and di_bullish and rsi[i] > 45 and rsi[i] < 70:
                new_signal = SIZE_ENTRY
            # Long: MACD crossover + 4h HMA bullish + ADX rising
            elif macd_hist[i] > 0 and macd_hist[i-1] <= 0 if not np.isnan(macd_hist[i-1]) else False and hma_bullish and adx[i] > adx[i-1] if not np.isnan(adx[i-1]) else adx[i] > 18:
                new_signal = SIZE_ENTRY
            
            # Short: 4h HMA bearish + 1h EMA bearish + MACD negative + ADX strong
            if hma_bearish and ema_bearish and macd_bearish and adx_strong and rsi_neutral_short:
                new_signal = -SIZE_ENTRY
            # Short: 4h HMA bearish + Price below EMA21 + DI bearish + RSI < 55
            elif hma_bearish and price_below_ema21 and di_bearish and rsi[i] > 30 and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
            # Short: MACD crossover down + 4h HMA bearish + ADX rising
            elif macd_hist[i] < 0 and macd_hist[i-1] >= 0 if not np.isnan(macd_hist[i-1]) else False and hma_bearish and adx[i] > adx[i-1] if not np.isnan(adx[i-1]) else adx[i] > 18:
                new_signal = -SIZE_ENTRY
        
        # === RANGE REGIME ENTRIES (CHOP > 55) ===
        elif in_range_regime:
            # Long: RSI oversold + Near BB lower + 4h HMA not strongly bearish
            if rsi_oversold and near_bb_lower and not (hma_bearish and adx_strong):
                new_signal = SIZE_ENTRY
            # Long: RSI < 35 + Price near BB lower + MACD histogram improving
            elif rsi[i] < 35 and near_bb_lower and macd_hist[i] > macd_hist[i-1] if not np.isnan(macd_hist[i-1]) else True:
                new_signal = SIZE_ENTRY
            # Long: Mean reversion from BB lower + RSI 25-40
            elif near_bb_lower and rsi[i] > 25 and rsi[i] < 40:
                new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + Near BB upper + 4h HMA not strongly bullish
            if rsi_overbought and near_bb_upper and not (hma_bullish and adx_strong):
                new_signal = -SIZE_ENTRY
            # Short: RSI > 65 + Price near BB upper + MACD histogram worsening
            elif rsi[i] > 65 and near_bb_upper and macd_hist[i] < macd_hist[i-1] if not np.isnan(macd_hist[i-1]) else True:
                new_signal = -SIZE_ENTRY
            # Short: Mean reversion from BB upper + RSI 60-75
            elif near_bb_upper and rsi[i] > 60 and rsi[i] < 75:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        # Only take high-confidence signals with reduced size
        else:
            # Long: Strong confluence (4h HMA + EMA + MACD + RSI)
            if hma_bullish and ema_bullish and macd_bullish and rsi[i] > 45 and rsi[i] < 60:
                new_signal = SIZE_HALF
            # Short: Strong confluence
            elif hma_bearish and ema_bearish and macd_bearish and rsi[i] > 40 and rsi[i] < 55:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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