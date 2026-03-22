#!/usr/bin/env python3
"""
Experiment #091: 15m Supertrend + 4h HMA Trend Filter + ADX + Volume
Hypothesis: 15m timeframe with strong 4h trend filter can capture intraday moves
while avoiding counter-trend trades that destroyed previous 15m strategies.

Why this might work (learning from failures):
- #079 (15m RSI pullback 4h HMA): Sharpe=-7.623 - RSI mean reversion fails on 15m
- #085 (15m RSI mean rev 4h HMA): Sharpe=-4.051 - same issue
- Key insight: 15m is TOO FAST for mean reversion, needs TREND FOLLOWING with HTF filter
- Supertrend works on 12h/4h (#083, #088), let's test on 15m WITH 4h HMA filter
- ADX ensures we only trade in trending conditions (avoid 15m chop)
- Volume confirmation filters false breakouts

Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.0*ATR (tighter for 15m).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_adx_vol_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # Initialize
    upper_band[:] = np.nan
    lower_band[:] = np.nan
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        # Calculate bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Determine supertrend value and direction
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1  # Start with long band
        else:
            # If previous supertrend was upper band
            if direction[i-1] == 1:
                if close[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            # If previous supertrend was lower band
            else:
                if close[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    mask = volume > 0
    vol_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        # st_direction: 1 = long signal (price above supertrend), -1 = short signal
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip detection (entry signal)
        st_flip_long = False
        st_flip_short = False
        if i > 0 and not np.isnan(st_direction[i-1]):
            st_flip_long = (st_direction[i] == 1) and (st_direction[i-1] == -1)
            st_flip_short = (st_direction[i] == -1) and (st_direction[i-1] == 1)
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending market (good for supertrend)
        # ADX > 25 = strong trend
        trending_market = adx[i] > 20
        strong_trend = adx[i] > 25
        
        # === VOLUME CONFIRMATION ===
        # vol_ratio > 0.55 = buying pressure, < 0.45 = selling pressure
        volume_bullish = vol_ratio[i] > 0.52
        volume_bearish = vol_ratio[i] < 0.48
        
        # === RSI MOMENTUM (light filter) ===
        rsi_momentum_long = rsi[i] > 40  # Not deeply oversold
        rsi_momentum_short = rsi[i] < 60  # Not deeply overbought
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Supertrend flip + 4h bullish + strong trend (strong signal)
        if st_flip_long and bull_trend_4h and strong_trend:
            if ema_bullish and volume_bullish:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + 4h bullish + trending (base signal)
        if new_signal == 0.0 and st_long and bull_trend_4h and trending_market:
            if ema_bullish or rsi_momentum_long:
                new_signal = SIZE_BASE
        
        # Path 3: Supertrend flip + 4h bullish (simpler, ensures trades)
        if new_signal == 0.0 and st_flip_long and bull_trend_4h:
            if trending_market or volume_bullish:
                new_signal = SIZE_BASE
        
        # Path 4: Supertrend long + EMA bullish + 4h bullish (fallback)
        if new_signal == 0.0 and st_long and ema_bullish and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Supertrend flip + 4h bearish + strong trend (strong signal)
        if st_flip_short and bear_trend_4h and strong_trend:
            if ema_bearish and volume_bearish:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + 4h bearish + trending (base signal)
        if new_signal == 0.0 and st_short and bear_trend_4h and trending_market:
            if ema_bearish or rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # Path 3: Supertrend flip + 4h bearish (simpler, ensures trades)
        if new_signal == 0.0 and st_flip_short and bear_trend_4h:
            if trending_market or volume_bearish:
                new_signal = -SIZE_BASE
        
        # Path 4: Supertrend short + EMA bearish + 4h bearish (fallback)
        if new_signal == 0.0 and st_short and ema_bearish and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 15m ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals