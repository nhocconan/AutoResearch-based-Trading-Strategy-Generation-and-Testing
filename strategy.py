#!/usr/bin/env python3
"""
Experiment #085: 15m Mean Reversion with 4h HMA Trend Filter + RSI Pullback
Hypothesis: 15m is too noisy for pure trend-following (see #073, #079 failures with Sharpe -4 to -7).
Instead, use 4h HMA for directional bias, then enter on 15m RSI pullbacks IN THE DIRECTION of 4h trend.
This is counter-trend on 15m but with-trend on 4h - captures dips in uptrends and rallies in downtrends.

Key insights from failed 15m experiments:
- #073 (CRSI mean reversion): Sharpe=-4.3, too many counter-trend trades
- #079 (RSI pullback + 4h HMA + 1h Supertrend): Sharpe=-7.6, over-filtered, 0 trades
- Solution: SIMPLER conditions, fewer filters, ensure trades on ALL symbols

Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 strong signals (discrete per Rule 4)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_4h_hma_v1"
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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === 4H TREND BIAS (HTF filter) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 15M TREND CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === SMA200 FILTER (long-term bias) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ZONES (mean reversion entries) ===
        # For longs in uptrend: buy dips when RSI pulls back but not crashed
        rsi_pullback_long = 35 <= rsi[i] <= 55
        # For shorts in downtrend: sell rallies when RSI bounces but not rocketing
        rsi_pullback_short = 45 <= rsi[i] <= 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] >= 40
        rsi_momentum_short = rsi[i] <= 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: 4h bullish + 15m EMA bullish + RSI pullback (primary trend continuation)
        if bull_trend_4h and ema_bullish and rsi_pullback_long and rsi_momentum_long:
            if above_sma200:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: 4h bullish + RSI oversold (strong mean reversion in uptrend)
        if bull_trend_4h and rsi[i] < 45 and rsi[i] > 30:
            if new_signal == 0.0:
                new_signal = SIZE_BASE
        
        # Path 3: Price above 4h HMA + above SMA200 + RSI healthy (trend continuation)
        if bull_trend_4h and above_sma200:
            if 40 <= rsi[i] <= 60:
                if new_signal == 0.0:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: 4h bearish + 15m EMA bearish + RSI pullback (primary trend continuation)
        if bear_trend_4h and ema_bearish and rsi_pullback_short and rsi_momentum_short:
            if below_sma200:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: 4h bearish + RSI overbought (strong mean reversion in downtrend)
        if bear_trend_4h and rsi[i] > 55 and rsi[i] < 70:
            if new_signal == 0.0:
                new_signal = -SIZE_BASE
        
        # Path 3: Price below 4h HMA + below SMA200 + RSI healthy (trend continuation)
        if bear_trend_4h and below_sma200:
            if 40 <= rsi[i] <= 60:
                if new_signal == 0.0:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6 - trailing ATR stop) ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals