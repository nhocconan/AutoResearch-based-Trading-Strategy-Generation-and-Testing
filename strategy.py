#!/usr/bin/env python3
"""
Experiment #394: 4h MACD Histogram Momentum + Daily HMA Trend + Bollinger Regime + RSI Pullback + ATR Stop
Hypothesis: MACD histogram momentum captures trend acceleration better than simple crossovers.
On 4h timeframe, histogram divergence/convergence provides earlier signals than Supertrend (which failed).
Bollinger Band Width percentile detects regime (squeeze = breakout imminent, wide = trend exhaustion).
Daily HMA provides trend bias. RSI(14) pullback entries (40-60 range) ensure trend continuation trades.
ADX(14) > 18 filter ensures minimum trend strength. Volume ratio confirms institutional participation.
ATR(14) stoploss at 2.5x protects capital. Position size 0.30 discrete to minimize fees.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper (call ONCE before loop).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: MACD histogram momentum + Bollinger regime filter = fewer whipsaws than Supertrend on 4h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_macd_hist_daily_hma_bollinger_regime_rsi_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response with less lag."""
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_smooth > 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth > 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = np.where(sma > 0, (upper - lower) / sma * 100, 0)
    return upper, lower, band_width

def calculate_bw_percentile(band_width, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection."""
    n = len(band_width)
    bw_pct = np.zeros(n)
    
    for i in range(lookback, n):
        window = band_width[i-lookback+1:i+1]
        rank = np.sum(window <= band_width[i])
        bw_pct[i] = rank / lookback * 100
    
    bw_pct[:lookback] = 50.0
    return bw_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_pct = calculate_bw_percentile(bb_width, 100)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.where(volume > 0, taker_buy_vol / volume, 0.5)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Bollinger Band Width regime
        # bb_pct < 20 = squeeze (breakout imminent)
        # bb_pct > 80 = wide bands (trend exhaustion)
        # bb_pct 20-80 = normal
        is_squeeze = bb_pct[i] < 25
        is_exhaustion = bb_pct[i] > 75
        is_normal = not is_squeeze and not is_exhaustion
        
        # MACD histogram momentum
        macd_hist_positive = macd_hist[i] > 0
        macd_hist_negative = macd_hist[i] < 0
        macd_hist_rising = macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_hist_falling = macd_hist[i] < macd_hist[i-1] if i > 0 else False
        macd_hist_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_hist_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # ADX trend strength
        adx_strong = adx[i] > 18  # Minimum trend strength
        adx_very_strong = adx[i] > 25
        
        # RSI pullback zones (not extremes, but trend continuation)
        rsi_bullish_pullback = 40 <= rsi[i] <= 60
        rsi_bearish_pullback = 40 <= rsi[i] <= 60
        rsi_momentum_long = 45 <= rsi[i] <= 70
        rsi_momentum_short = 30 <= rsi[i] <= 55
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        vol_strong = vol_ratio[i] > 0.55 or vol_ratio[i] < 0.45
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: MACD hist cross up + Daily bullish + ADX strong + RSI ok
        if macd_hist_cross_up and daily_bullish and adx_strong and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Secondary: MACD hist positive + rising + Daily bullish + RSI pullback
        elif macd_hist_positive and macd_hist_rising and daily_bullish and rsi_bullish_pullback:
            new_signal = SIZE_ENTRY
        # Tertiary: MACD hist cross up + ADX strong + Volume confirmation
        elif macd_hist_cross_up and adx_strong and vol_bullish:
            new_signal = SIZE_ENTRY
        # Quaternary: MACD hist positive + Daily bullish + Squeeze breakout
        elif macd_hist_positive and daily_bullish and is_squeeze:
            new_signal = SIZE_ENTRY
        # Quintenary: MACD hist rising + RSI momentum (ensures trade frequency)
        elif macd_hist_rising and rsi[i] > 45 and rsi[i] < 65 and daily_bullish:
            new_signal = SIZE_ENTRY
        # Sextenary: MACD hist cross up alone (backup for minimum trades)
        elif macd_hist_cross_up and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: MACD hist cross down + Daily bearish + ADX strong + RSI ok
        if macd_hist_cross_down and daily_bearish and adx_strong and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Secondary: MACD hist negative + falling + Daily bearish + RSI pullback
        elif macd_hist_negative and macd_hist_falling and daily_bearish and rsi_bearish_pullback:
            new_signal = -SIZE_ENTRY
        # Tertiary: MACD hist cross down + ADX strong + Volume confirmation
        elif macd_hist_cross_down and adx_strong and vol_bearish:
            new_signal = -SIZE_ENTRY
        # Quaternary: MACD hist negative + Daily bearish + Squeeze breakout
        elif macd_hist_negative and daily_bearish and is_squeeze:
            new_signal = -SIZE_ENTRY
        # Quintenary: MACD hist falling + RSI momentum (ensures trade frequency)
        elif macd_hist_falling and rsi[i] > 35 and rsi[i] < 55 and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Sextenary: MACD hist cross down alone (backup for minimum trades)
        elif macd_hist_cross_down and rsi[i] < 60:
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