#!/usr/bin/env python3
"""
Experiment #229: 15m Donchian Breakout + 4h HMA Trend + 1h RSI + ADX Filter

Hypothesis: 15m timeframe captures intraday momentum moves while 4h HMA provides
stable higher-timeframe bias. Donchian breakouts (20-period) on 15m capture
momentum moves, 4h HMA filters counter-trend trades, 1h RSI confirms momentum,
and ADX > 15 (lower threshold for 15m) ensures sufficient trade count.

Why 15m might work:
- 15m = 96 bars/day, captures intraday swings without 5m noise
- Donchian breakouts work on multiple timeframes (Turtle Trading principle)
- 4h HMA filter (4x slower) prevents counter-trend breakouts
- 1h RSI adds momentum confirmation without look-ahead
- ADX > 15 (not 25) ensures we get trades on 15m timeframe
- Conservative sizing (0.30) controls drawdown

Learning from failures:
- #217 (15m KAMA): Sharpe=-1.931 - KAMA whipsaws in choppy 15m markets
- #223 (15m mean rev): Sharpe=-2.849 - mean reversion fails on crypto 15m
- #224 (30m Supertrend): Sharpe=0.000 - 0 trades, conditions too strict
- #219 (1h Fisher): Sharpe=0.000 - 0 trades, Fisher too rare on crypto
- Breakout + trend filter works better than pure mean reversion
- Need LOWER ADX threshold on 15m to ensure trade count

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper + 1h RSI for momentum (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing (tighter for 15m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_donchian_4h_hma_1h_rsi_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values with first valid value
    upper[:period-1] = upper[period-1]
    lower[:period-1] = lower[period-1]
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 15m to ensure trades)
        trend_strength = adx[i] > 15
        
        # === DONCHIAN BREAKOUT ===
        # Price breaking above upper = bullish breakout
        # Price breaking below lower = bearish breakout
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === RSI MOMENTUM (1h aligned) ===
        # RSI > 45 = bullish momentum (not too strict)
        # RSI < 55 = bearish momentum
        rsi_bullish = rsi_1h_aligned[i] > 45
        rsi_bearish = rsi_1h_aligned[i] < 55
        
        # === EMA CONFIRMATION ===
        # EMA21 > EMA50 = bullish trend structure
        # EMA21 < EMA50 = bearish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === SMA200 FILTER ===
        # Price above SMA200 = long-term bullish
        # Price below SMA200 = long-term bearish
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + ADX trending + Donchian breakout + (RSI OR EMA confirmation)
        # Using OR logic to ensure we get enough trades
        if bull_trend_4h and trend_strength and breakout_long:
            # Need at least one momentum confirmation (loose conditions)
            if rsi_bullish or ema_bullish or above_sma200:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + ADX trending + Donchian breakout + (RSI OR EMA confirmation)
        if bear_trend_4h and trend_strength and breakout_short:
            # Need at least one momentum confirmation (loose conditions)
            if rsi_bearish or ema_bearish or below_sma200:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = int(np.sign(new_signal))
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = int(np.sign(new_signal))
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals