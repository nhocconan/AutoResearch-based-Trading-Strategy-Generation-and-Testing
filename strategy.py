#!/usr/bin/env python3
"""
Experiment #348: 1d Daily Strategy with 1w HMA Bias + RSI + ADX Filter

Hypothesis: Daily timeframe with weekly trend bias provides stable directional
filter while RSI entries on daily generate sufficient trade frequency. ADX
filter avoids choppy periods where both trend and mean-reversion fail.

Key design choices:
1. 1w HMA(21) for ultra-stable trend bias (only flips in major regime changes)
2. RSI(14) with loose thresholds (<40 long, >60 short) for enough trades
3. ADX(14) > 18 filter to avoid dead choppy markets
4. Position size 0.25 discrete, ATR(14) stoploss at 2.5x
5. Asymmetric: prefer longs when 1w HMA bullish, shorts when bearish

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_adx_1w_hma_bias_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high_s - close_s.shift(1)).values
    tr3 = np.abs(low_s - close_s.shift(1)).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
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
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * plus_dm_smooth / np.maximum(tr_smooth, 1e-10)
    di_minus = 100 * minus_dm_smooth / np.maximum(tr_smooth, 1e-10)
    
    # DX
    di_sum = di_plus + di_minus
    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_sum, 1e-10)
    
    # ADX = smoothed DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === 1w HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === ADX TREND STRENGTH FILTER ===
        trending_market = adx[i] > 18  # Loose threshold for more trades
        
        # === RSI ENTRY SIGNALS (loose thresholds for trade frequency) ===
        rsi_oversold = rsi[i] < 40  # Long entry
        rsi_overbought = rsi[i] > 60  # Short entry
        rsi_extreme_long = rsi[i] < 30  # Strong long
        rsi_extreme_short = rsi[i] > 70  # Strong short
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG entries: RSI oversold + 1w HMA bullish bias OR extreme oversold
        if rsi_extreme_long:
            # Very oversold - enter regardless of trend (mean reversion)
            new_signal = SIZE
        elif rsi_oversold and bull_trend_1w and trending_market:
            # Oversold + bullish trend + trending market
            new_signal = SIZE
        elif rsi_oversold and bull_trend_1w:
            # Oversold + bullish trend (ADX filter relaxed)
            new_signal = SIZE * 0.6  # Smaller position
        
        # SHORT entries: RSI overbought + 1w HMA bearish bias OR extreme overbought
        if rsi_extreme_short:
            # Very overbought - enter regardless of trend (mean reversion)
            new_signal = -SIZE
        elif rsi_overbought and bear_trend_1w and trending_market:
            # Overbought + bearish trend + trending market
            new_signal = -SIZE
        elif rsi_overbought and bear_trend_1w:
            # Overbought + bearish trend (ADX filter relaxed)
            new_signal = -SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1w trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w and adx[i] > 25:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w and adx[i] > 25:
                new_signal = 0.0
        
        # === RSI REVERSAL EXIT ===
        # Exit long when RSI becomes overbought, exit short when oversold
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 30:
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