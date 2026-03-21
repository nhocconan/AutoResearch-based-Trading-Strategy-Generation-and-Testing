#!/usr/bin/env python3
"""
Experiment #427: 15m HMA Trend + 4h HMA Bias + RSI Pullback + ADX Regime + ATR Stop
Hypothesis: 15m timeframe captures more frequent mean-reversion opportunities within 
the 4h trend direction. Using HMA for smoother trend detection, RSI pullback entries
(oversold in uptrend, overbought in downtrend), and ADX filter to avoid choppy markets.
Key insight: 15m needs faster indicators than 12h but still requires HTF trend filter
to avoid whipsaws. Multiple entry paths ensure >=10 trades while ADX>20 filters noise.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.0*ATR for 15m timeframe (tighter than 12h).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_4h_bias_rsi_pullback_adx_regime_atr_v1"
timeframe = "15m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and ADX
    plus_di = np.where(tr_s > 0, 100 * plus_dm_s / tr_s, 0)
    minus_di = np.where(tr_s > 0, 100 * minus_dm_s / tr_s, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    hma_15m = calculate_hma(close, 16)
    ema_21 = calculate_ema(close, 21)
    sma_50 = calculate_sma(close, 50)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        trend_15m_bullish = hma_15m[i] > ema_21[i] and close[i] > sma_50[i]
        trend_15m_bearish = hma_15m[i] < ema_21[i] and close[i] < sma_50[i]
        
        # ADX regime filter (ADX > 20 = trending, ADX < 20 = ranging)
        is_trending = adx[i] > 20
        is_ranging = adx[i] <= 20
        
        # DI crossover for momentum
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # RSI levels for pullback entries
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 60
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bullish + RSI pullback + ADX trending (primary)
        if trend_4h_bullish and rsi_oversold and is_trending:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + 15m bullish + RSI neutral + DI bullish
        elif trend_4h_bullish and trend_15m_bullish and rsi_neutral_long and di_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: 4h bullish + RSI momentum + price > HMA15
        elif trend_4h_bullish and rsi_momentum_long and close[i] > hma_15m[i]:
            new_signal = SIZE_ENTRY
        # Path 4: Ranging market + RSI oversold + 4h bullish (mean reversion)
        elif is_ranging and rsi_oversold and trend_4h_bullish and rsi[i] < 30:
            new_signal = SIZE_ENTRY
        # Path 5: DI bullish crossover + 4h bullish + RSI > 40
        elif di_bullish and trend_4h_bullish and rsi[i] > 40 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bearish + RSI pullback + ADX trending (primary)
        if trend_4h_bearish and rsi_overbought and is_trending:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + 15m bearish + RSI neutral + DI bearish
        elif trend_4h_bearish and trend_15m_bearish and rsi_neutral_short and di_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h bearish + RSI momentum + price < HMA15
        elif trend_4h_bearish and rsi_momentum_short and close[i] < hma_15m[i]:
            new_signal = -SIZE_ENTRY
        # Path 4: Ranging market + RSI overbought + 4h bearish (mean reversion)
        elif is_ranging and rsi_overbought and trend_4h_bearish and rsi[i] > 70:
            new_signal = -SIZE_ENTRY
        # Path 5: DI bearish crossover + 4h bearish + RSI < 60
        elif di_bearish and trend_4h_bearish and rsi[i] < 60 and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest for 15m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest for 15m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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