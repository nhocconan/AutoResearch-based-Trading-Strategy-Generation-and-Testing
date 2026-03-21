#!/usr/bin/env python3
"""
Experiment #037: 15m Supertrend + 4h HMA Regime + RSI Pullback + Volume
Hypothesis: 15m timeframe captures intraday swings with 4h HMA as major trend filter.
Supertrend provides clear direction signals, RSI pullbacks give entry timing within trend.
Multiple entry triggers (Supertrend flip, RSI extreme + trend, breakout) ensure ≥10 trades.
Position sizing 0.28 with 2.5x ATR stoploss protects against crashes while capturing trends.
Key insight from #035: relaxed thresholds (RSI 30/70 not 20/80) ensure trades generate.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_vol_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_ema(close, period=21):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # 15m EMAs for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime) - relaxed for more trades
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        fourh_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        fourh_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry trigger)
        st_flip_long = st_direction[i] == 1 and i > 0 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and i > 0 and st_direction[i-1] == 1
        
        # EMA trend confirmation
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        
        # RSI extremes (relaxed for more trades - Rule 9)
        rsi_oversold = rsi[i] < 35  # Was 30, relaxed for more trades
        rsi_overbought = rsi[i] > 65  # Was 70, relaxed for more trades
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 55
        
        # RSI momentum (rising/falling)
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation (relaxed)
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        # Price position vs EMA21
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Supertrend flip long (strongest signal)
        if st_flip_long:
            new_signal = SIZE
        # Trigger 2: Supertrend long + 4h bullish + RSI bullish
        elif st_long and fourh_bullish and rsi_bullish:
            new_signal = SIZE
        # Trigger 3: Supertrend long + EMA trend + RSI rising + volume
        elif st_long and ema_trend_long and rsi_rising and vol_confirm:
            new_signal = SIZE
        # Trigger 4: 4h bullish + Supertrend long + price above EMA (trend continuation)
        elif fourh_bullish and st_long and price_above_ema:
            new_signal = SIZE
        # Trigger 5: RSI oversold + Supertrend long (pullback entry in uptrend)
        elif rsi_oversold and st_long:
            new_signal = SIZE
        # Trigger 6: Supertrend long + EMA trend (simple trend follow)
        elif st_long and ema_trend_long and vol_confirm:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Supertrend flip short (strongest signal)
        if st_flip_short:
            new_signal = -SIZE
        # Trigger 2: Supertrend short + 4h bearish + RSI bearish
        elif st_short and fourh_bearish and rsi_bearish:
            new_signal = -SIZE
        # Trigger 3: Supertrend short + EMA trend + RSI falling + volume
        elif st_short and ema_trend_short and rsi_falling and vol_confirm:
            new_signal = -SIZE
        # Trigger 4: 4h bearish + Supertrend short + price below EMA (trend continuation)
        elif fourh_bearish and st_short and price_below_ema:
            new_signal = -SIZE
        # Trigger 5: RSI overbought + Supertrend short (pullback entry in downtrend)
        elif rsi_overbought and st_short:
            new_signal = -SIZE
        # Trigger 6: Supertrend short + EMA trend (simple trend follow)
        elif st_short and ema_trend_short and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if trailing_stop > 0 and close[i] < trailing_stop:
                    new_signal = 0.0
                # Take partial profit at 3R
                if close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if trailing_stop > 0 and close[i] > trailing_stop:
                    new_signal = 0.0
                # Take partial profit at 3R
                if close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals