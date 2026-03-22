#!/usr/bin/env python3
"""
Experiment #075: 1h Z-Score Mean Reversion with 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 1h timeframe captures intraday mean reversion opportunities while 4h HMA provides
trend bias to avoid counter-trend trades. Z-score(20) extremes (<-2 or >+2) signal overextended
moves likely to revert. Volume confirmation (taker_buy_ratio > 0.55 for longs) ensures genuine
interest. ATR filter avoids trading during extreme volatility spikes.

Why this might work:
1. Z-score mean reversion has proven edge in crypto (70%+ win rate at extremes)
2. 4h HMA trend filter prevents dangerous counter-trend entries during strong trends
3. Volume confirmation filters out fake breakouts/reversals
4. ATR volatility filter avoids whipsaw during panic pumps/dumps
5. Asymmetric sizing: smaller positions in bear regime (4h HMA sloping down)

Key improvements vs failed strategies:
- Simpler entry logic (Z-score + trend + volume) = more trades
- No complex regime switching that causes 0 trades
- Discrete position sizes (0.20, 0.30) to minimize fee churn
- Stoploss at 2.5*ATR with trailing for winners

Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20 base, 0.30 strong signal, max 0.35
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_zscore_4h_hma_vol_atr_v1"
timeframe = "1h"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs rolling mean."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values

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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio."""
    ratio = taker_buy_volume / (volume + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # ATR ratio for volatility filter (ATR(7)/ATR(30))
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_MAX = 0.35
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
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
        
        if np.isnan(zscore[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (compare to 3 bars ago)
        hma_slope_bull = i >= 3 and not np.isnan(hma_4h_aligned[i-3]) and hma_4h_aligned[i] > hma_4h_aligned[i-3]
        hma_slope_bear = i >= 3 and not np.isnan(hma_4h_aligned[i-3]) and hma_4h_aligned[i] < hma_4h_aligned[i-3]
        
        # EMA alignment on 1h
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === VOLATILITY FILTER ===
        # Avoid trading during extreme vol spikes (ATR ratio > 2.5)
        normal_vol = atr_ratio[i] < 2.5
        low_vol = atr_ratio[i] < 1.5
        
        # === Z-SCORE MEAN REVERSION SIGNALS ===
        # Z-score <-2 = oversold (potential long)
        # Z-score >+2 = overbought (potential short)
        zscore_oversold = zscore[i] < -2.0
        zscore_overbought = zscore[i] > 2.0
        zscore_extreme_oversold = zscore[i] < -2.5
        zscore_extreme_overbought = zscore[i] > 2.5
        
        # === VOLUME CONFIRMATION ===
        # For longs: want buying pressure (taker buy ratio > 0.55)
        # For shorts: want selling pressure (taker buy ratio < 0.45)
        vol_buying = vol_ratio[i] > 0.55
        vol_selling = vol_ratio[i] < 0.45
        vol_neutral = 0.45 <= vol_ratio[i] <= 0.55
        
        # === RSI FILTER ===
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === PRICE VS EMA ===
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Z-score oversold + bull trend + volume confirmation
        if bull_trend_4h and zscore_oversold and normal_vol:
            if vol_buying or rsi_oversold:
                if hma_slope_bull:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Extreme Z-score oversold (stronger signal, less volume req)
        if zscore_extreme_oversold and normal_vol:
            if bull_trend_4h or ema_bullish:
                new_signal = SIZE_BASE
        
        # Path 3: RSI oversold + bull trend + price near EMA support
        if bull_trend_4h and rsi_oversold and price_above_ema21:
            if vol_buying:
                new_signal = SIZE_BASE
        
        # Path 4: EMA bullish + pullback to EMA21 + neutral Z-score
        if ema_bullish and bull_trend_4h:
            if -1.0 < zscore[i] < 0.5 and price_above_ema21:
                if vol_buying:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Z-score overbought + bear trend + volume confirmation
        if bear_trend_4h and zscore_overbought and normal_vol:
            if vol_selling or rsi_overbought:
                if hma_slope_bear:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Extreme Z-score overbought (stronger signal, less volume req)
        if zscore_extreme_overbought and normal_vol:
            if bear_trend_4h or ema_bearish:
                new_signal = -SIZE_BASE
        
        # Path 3: RSI overbought + bear trend + price below EMA resistance
        if bear_trend_4h and rsi_overbought and price_below_ema21:
            if vol_selling:
                new_signal = -SIZE_BASE
        
        # Path 4: EMA bearish + rally to EMA21 + neutral Z-score
        if ema_bearish and bear_trend_4h:
            if -0.5 < zscore[i] < 1.0 and price_below_ema21:
                if vol_selling:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals