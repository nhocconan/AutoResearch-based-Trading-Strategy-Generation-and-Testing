#!/usr/bin/env python3
"""
Experiment #006: 1d Weekly HMA Trend + ATR Volatility Breakout + RSI Filter
Hypothesis: Daily timeframe with weekly trend filter reduces whipsaws. ATR volatility expansion
captures true breakouts vs fake moves. RSI(14) filter avoids chasing extended moves.
This should work across BTC/ETH/SOL because it adapts to regime (trend vs range).

Key innovations:
1. Weekly HMA(21) for primary trend bias (loaded via mtf_data ONCE)
2. ATR(14) expansion ratio > 1.5 confirms real breakout momentum
3. RSI(14) between 35-65 for entry (not oversold/overbought = trend continuation)
4. Bollinger Band width percentile for regime detection
5. Trailing stop at 2.5*ATR protects capital in 2022-style crashes
6. Position sizing: 0.30 base, discrete levels to minimize fee churn

Timeframe: 1d (REQUIRED for this experiment), HTF: 1w via mtf_data helper.
This should generate 20-50 trades/year on daily data (fewer but higher quality).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_atr_vol_breakout_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:period] = np.nan
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0.0)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    
    # ATR ratio for volatility expansion
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Volume MA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # EMA for trend confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.30
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Bollinger bandwidth percentile for regime detection
    bb_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / len(x[:-1]) * 100 if len(x) > 1 else 50
    ).values
    bb_percentile[np.isnan(bb_percentile)] = 50.0
    
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
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # Regime detection
        range_regime = bb_percentile[i] < 40  # Low bandwidth = ranging
        trend_regime = bb_percentile[i] > 60  # High bandwidth = trending
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # ATR volatility expansion (confirms real breakout)
        vol_expansion = atr_ratio[i] > 1.3
        
        # RSI filter (not extreme = trend continuation)
        rsi_neutral = 35 < rsi[i] < 65
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        # Price position vs EMA
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        price_above_ema50 = close[i] > ema_50[i]
        price_below_ema50 = close[i] < ema_50[i]
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_sma[i] * 1.2
        
        # Price vs Bollinger Bands
        price_break_upper = close[i] > bb_upper[i] * 0.998
        price_break_lower = close[i] < bb_lower[i] * 1.002
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bull trend + ATR expansion + RSI bullish + volume
        if bull_trend and vol_expansion and rsi_bullish and vol_above_avg:
            if price_above_ema21 and price_above_ema50:
                new_signal = SIZE_MAX
            elif price_above_ema21 and strong_trend:
                new_signal = SIZE_BASE
        
        # Secondary: Range regime + price near lower BB + weekly bull trend
        elif range_regime and price_break_lower and bull_trend and rsi[i] < 40:
            new_signal = SIZE_BASE
        
        # Tertiary: Trend regime + breakout above BB + weekly bull
        elif trend_regime and price_break_upper and bull_trend and rsi_neutral:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Weekly bear trend + ATR expansion + RSI bearish + volume
        if bear_trend and vol_expansion and rsi_bearish and vol_above_avg:
            if price_below_ema21 and price_below_ema50:
                new_signal = -SIZE_MAX
            elif price_below_ema21 and strong_trend:
                new_signal = -SIZE_BASE
        
        # Secondary: Range regime + price near upper BB + weekly bear trend
        elif range_regime and price_break_upper and bear_trend and rsi[i] > 60:
            new_signal = -SIZE_BASE
        
        # Tertiary: Trend regime + breakdown below BB + weekly bear
        elif trend_regime and price_break_lower and bear_trend and rsi_neutral:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals