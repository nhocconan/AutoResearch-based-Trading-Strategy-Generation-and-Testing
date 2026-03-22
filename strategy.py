#!/usr/bin/env python3
"""
Experiment #027: 1h Mean Reversion with 4h Trend Bias + Vol Spike Filter
Hypothesis: 1h timeframe captures short-term mean reversion opportunities while 4h HMA provides trend bias.
Key insight: Trend-following failed repeatedly (exp#015-026 all negative Sharpe). Mean reversion works better in bear/range markets (2025 test).
Strategy: 4h HMA for regime bias, 1h RSI(3) extremes for entry, BB position confirmation, ATR vol spike filter.
Why this might work: RSI(3) extremes catch oversold/overbought bounces, 4h HMA avoids counter-trend trades, vol spike filter catches panic reversals.
Position sizing: 0.25 discrete, stoploss at 2.5*ATR, asymmetric (smaller size in bear regime).
Must generate 10+ trades on train, 3+ on test - RSI thresholds loosened (10/90 not 5/95).
Timeframe: 1h (REQUIRED for exp#027), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_meanrev_4h_hma_rsi3_bb_v1"
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
    bb_width = (upper - lower) / (sma + 1e-10)
    bb_position = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, bb_width, bb_position

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(er_period)).values
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(len(close))
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1))**2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_3 = calculate_rsi(close, 3)  # Fast RSI for mean reversion
    rsi_14 = calculate_rsi(close, 14)  # Standard RSI for confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_width, bb_position = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR ratio for vol spike detection
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BULL = 0.30  # Larger size in bull regime
    SIZE_BEAR = 0.20  # Smaller size in bear regime
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
        
        if np.isnan(rsi_3[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_position[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_regime_4h = close[i] > hma_4h_aligned[i]
        bear_regime_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI(3) extremes for mean reversion - LOOSENED for more trades
        rsi3_oversold = rsi_3[i] < 15  # Was 10, loosened
        rsi3_overbought = rsi_3[i] > 85  # Was 90, loosened
        
        # RSI(14) confirmation
        rsi14_oversold = rsi_14[i] < 35
        rsi14_overbought = rsi_14[i] > 65
        
        # Bollinger Band position
        bb_low = bb_position[i] < 0.15  # Near lower band
        bb_high = bb_position[i] > 0.85  # Near upper band
        
        # Vol spike filter - catch panic reversals
        vol_spike = atr_ratio[i] > 1.5  # ATR(7) > 1.5x ATR(30)
        vol_normal = atr_ratio[i] < 1.3
        
        # KAMA slope for trend confirmation
        kama_slope_up = False
        kama_slope_down = False
        if i >= 3:
            kama_slope_up = kama[i] > kama[i-3]
            kama_slope_down = kama[i] < kama[i-3]
        
        # Price distance from KAMA
        kama_distance = (close[i] - kama[i]) / (kama[i] + 1e-10)
        far_below_kama = kama_distance < -0.03  # 3% below KAMA
        far_above_kama = kama_distance > 0.03  # 3% above KAMA
        
        # Select position size based on regime
        current_size = SIZE_BULL if bull_regime_4h else SIZE_BEAR
        
        new_signal = 0.0
        
        # === LONG ENTRIES (Mean Reversion) ===
        # Primary: RSI(3) oversold + BB low + vol spike (panic bounce)
        if rsi3_oversold and bb_low and vol_spike:
            # Only long if not in strong bear regime
            if bull_regime_4h or (bear_regime_4h and above_200):
                new_signal = current_size
        
        # Secondary: RSI(3) oversold + RSI(14) confirmation + 4h bull
        elif rsi3_oversold and rsi14_oversold and bull_regime_4h:
            new_signal = current_size
        
        # Tertiary: Far below KAMA + 4h bull (stretch trade)
        elif far_below_kama and bull_regime_4h and vol_normal:
            new_signal = SIZE_HALF
        
        # Quaternary: BB low + 4h bull + above 200 SMA
        elif bb_low and bull_regime_4h and above_200:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (Mean Reversion) ===
        # Primary: RSI(3) overbought + BB high + vol spike (panic top)
        if rsi3_overbought and bb_high and vol_spike:
            # Only short if not in strong bull regime
            if bear_regime_4h or (bull_regime_4h and below_200):
                new_signal = -current_size
        
        # Secondary: RSI(3) overbought + RSI(14) confirmation + 4h bear
        elif rsi3_overbought and rsi14_overbought and bear_regime_4h:
            new_signal = -current_size
        
        # Tertiary: Far above KAMA + 4h bear (stretch trade)
        elif far_above_kama and bear_regime_4h and vol_normal:
            new_signal = -SIZE_HALF
        
        # Quaternary: BB high + 4h bear + below 200 SMA
        elif bb_high and bear_regime_4h and below_200:
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