#!/usr/bin/env python3
"""
Experiment #013: 15m Multi-Timeframe Mean Reversion with 4h Trend Filter
Hypothesis: 15m timeframe captures short-term mean reversion opportunities while 4h HMA provides trend bias.
Key insight: Previous 15m strategies failed due to overly complex regime detection or too-strict entry conditions.
This strategy uses: 4h HMA for trend direction, 15m RSI(7) for pullback entries, Bollinger Bands for oversold/overbought zones.
Position sizing: 0.25-0.30 discrete levels with 2.5*ATR stoploss to control drawdown.
Timeframe: 15m (REQUIRED for exp#013), HTF: 4h via mtf_data helper.
Why this might work: 15m has more trade opportunities than 1h/4h, RSI(7) is more sensitive than RSI(14) for mean reversion.
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_rsi_bb_v1"
timeframe = "15m"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_momentum(close, period=10):
    """Calculate price momentum (ROC)."""
    mom = np.zeros(len(close))
    mom[:] = np.nan
    for i in range(period, len(close)):
        if close[i-period] != 0:
            mom[i] = (close[i] - close[i-period]) / close[i-period] * 100
    return mom

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
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    zscore = calculate_zscore(close, 20)
    momentum = calculate_momentum(close, 10)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # BB Width for regime detection
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width_sma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        bull_trend_15m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_15m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_oversold_long = rsi_7[i] < 35  # More sensitive on 15m
        rsi_overbought_short = rsi_7[i] > 65
        rsi_neutral = 35 <= rsi_7[i] <= 65
        
        # Bollinger Band conditions
        at_bb_lower = close[i] <= bb_lower[i] * 1.005
        at_bb_upper = close[i] >= bb_upper[i] * 0.995
        near_bb_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_mid[i]) * 0.3
        
        # Z-score filter - avoid extreme entries
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_neutral = abs(zscore[i]) < 2.0
        
        # Momentum confirmation
        mom_positive = not np.isnan(momentum[i]) and momentum[i] > -2.0
        mom_negative = not np.isnan(momentum[i]) and momentum[i] < 2.0
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        # Volume confirmation (if available)
        volume_confirmed = True
        if 'volume' in prices.columns:
            vol_sma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
            if not np.isnan(vol_sma[i]):
                volume_confirmed = prices['volume'].values[i] > vol_sma[i] * 0.8
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish or neutral) ===
        if bull_trend_4h or (not bear_trend_4h and above_200):
            # Primary: RSI oversold + BB lower band + 4h trend
            if rsi_oversold_long and at_bb_lower and zscore_neutral:
                new_signal = SIZE_BASE
            
            # Secondary: Pullback to EMA21 in uptrend
            elif close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99 and bull_trend_15m and rsi_7[i] < 50:
                new_signal = SIZE_BASE
            
            # Tertiary: Higher low with RSI bounce
            elif higher_low and rsi_7[i] > 30 and rsi_7[i] < 55 and mom_positive:
                new_signal = SIZE_HALF
            
            # Quaternary: Z-score mean reversion in uptrend
            elif zscore_oversold and bull_trend_4h and above_200:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish or neutral) ===
        elif bear_trend_4h or (not bull_trend_4h and below_200):
            # Primary: RSI overbought + BB upper band + 4h trend
            if rsi_overbought_short and at_bb_upper and zscore_neutral:
                new_signal = -SIZE_BASE
            
            # Secondary: Bounce to EMA21 in downtrend
            elif close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01 and bear_trend_15m and rsi_7[i] > 50:
                new_signal = -SIZE_BASE
            
            # Tertiary: Lower high with RSI rejection
            elif lower_high and rsi_7[i] < 70 and rsi_7[i] > 45 and mom_negative:
                new_signal = -SIZE_HALF
            
            # Quaternary: Z-score mean reversion in downtrend
            elif zscore_overbought and bear_trend_4h and below_200:
                new_signal = -SIZE_HALF
        
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