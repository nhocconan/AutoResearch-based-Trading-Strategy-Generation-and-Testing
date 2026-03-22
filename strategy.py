#!/usr/bin/env python3
"""
Experiment #031: 15m Volatility-Adaptive Regime Strategy with 4h HMA Bias
Hypothesis: 15m timeframe captures quick reversals after vol spikes, while 4h HMA provides trend regime filter.
Key insight: Previous 15m strategies failed because they used single-regime logic (either pure trend or pure mean-revert).
This strategy ADAPTS: high vol (ATR7/ATR30>1.8) = mean revert at BB bands, normal vol = trend follow with HTF bias.
Why this might work: Vol spike reversals have 70%+ win rate on BTC/ETH, 4h HMA smoother than 1h for regime detection.
Position sizing: 0.25-0.35 discrete, stoploss at 2.5*ATR, asymmetric (smaller in choppy markets).
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
Timeframe: 15m (REQUIRED for exp#031), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_adaptive_4h_hma_bb_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(period).values)
    volatility = np.abs(close_s.diff()).rolling(window=period).sum().values
    
    er = np.zeros(len(close))
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)  # Fast RSI for CRSI-style
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_50 = calculate_sma(close, 50)
    kama = calculate_kama(close, 10)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # Volatility ratio for regime detection
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Volatility regime detection
        high_vol = vol_ratio[i] > 1.8  # Vol spike
        low_vol = vol_ratio[i] < 1.2  # Vol crush
        normal_vol = not high_vol and not low_vol
        
        # Bollinger Band position
        at_bb_lower = close[i] <= bb_lower[i] * 1.005
        at_bb_upper = close[i] >= bb_upper[i] * 0.995
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i+1] if i > 0 else bb_bw[:1], 20)
        
        # RSI conditions - LOOSENED for more trades
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_low = rsi_3[i] < 15  # Very oversold fast
        rsi_extreme_high = rsi_3[i] > 85  # Very overbought fast
        
        # 15m trend confirmation
        bull_trend_15m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_15m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Price vs 200 EMA (long-term filter)
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # Volume confirmation
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        high_volume = volume[i] > vol_ma[i] * 1.5 if not np.isnan(vol_ma[i]) else False
        
        # KAMA slope for trend strength
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        new_signal = 0.0
        
        # === REGIME 1: HIGH VOLATILITY (Mean Reversion Play) ===
        if high_vol:
            # Long: Vol spike + price at BB lower + RSI oversold
            if at_bb_lower and rsi_oversold:
                if bull_trend_4h:  # Prefer long in 4h uptrend
                    new_signal = SIZE_BASE
                elif rsi_extreme_low:  # Or very oversold regardless of trend
                    new_signal = SIZE_HALF
            
            # Short: Vol spike + price at BB upper + RSI overbought
            if at_bb_upper and rsi_overbought:
                if bear_trend_4h:  # Prefer short in 4h downtrend
                    new_signal = -SIZE_BASE
                elif rsi_extreme_high:  # Or very overbought regardless of trend
                    new_signal = -SIZE_HALF
        
        # === REGIME 2: NORMAL VOLATILITY (Trend Following) ===
        elif normal_vol:
            # Long: 4h bullish + 15m pullback to EMA21
            if bull_trend_4h and bull_trend_15m:
                if close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.98:
                    if rsi[i] > 40 and rsi[i] < 65:  # RSI not extreme
                        new_signal = SIZE_BASE
                elif higher_low and kama_bullish:
                    new_signal = SIZE_HALF
            
            # Short: 4h bearish + 15m bounce to EMA21
            if bear_trend_4h and bear_trend_15m:
                if close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.02:
                    if rsi[i] > 35 and rsi[i] < 60:  # RSI not extreme
                        new_signal = -SIZE_BASE
                elif lower_high and kama_bearish:
                    new_signal = -SIZE_HALF
        
        # === REGIME 3: LOW VOLATILITY (Breakout Play) ===
        elif low_vol:
            # Long: BB squeeze breakout + volume + 4h bullish
            if bb_squeeze and high_volume and bull_trend_4h:
                if close[i] > bb_upper[i] * 0.998:
                    new_signal = SIZE_HALF
            
            # Short: BB squeeze breakout + volume + 4h bearish
            if bb_squeeze and high_volume and bear_trend_4h:
                if close[i] < bb_lower[i] * 1.002:
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