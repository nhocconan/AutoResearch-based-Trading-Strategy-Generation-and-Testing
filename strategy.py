#!/usr/bin/env python3
"""
Experiment #024: 1d KAMA Trend with 1w HMA Regime Filter
Hypothesis: Daily timeframe captures major trends with fewer whipsaws. Weekly HMA provides robust regime bias.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - fast in trends, slow in ranges.
Key insight: Previous 1d strategies failed due to overly strict conditions. This uses multiple entry triggers.
Entry conditions LOOSENED: RSI 35-65 range, any of 3 entry signals can trigger.
Position sizing: 0.30 discrete levels with 2.5*ATR trailing stop.
Timeframe: 1d (REQUIRED for exp#024), HTF: 1w via mtf_data helper.
Why this might work: KAMA reduces whipsaw in choppy markets, 1w HMA smoother than 1d for regime.
Must generate 10+ trades on train, 3+ on test - multiple entry paths ensure this.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_1w_hma_rsi_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average - adapts to market efficiency.
    Fast in trending markets, slow in ranging markets.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / (volatility[mask] + 1e-10)
    er[:period] = np.nan
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    kama_10 = calculate_kama(close, 10, 2, 30)
    kama_30 = calculate_kama(close, 30, 2, 30)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - main regime filter
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d trend confirmation via KAMA
        bull_trend_1d = kama_10[i] > kama_30[i]
        bear_trend_1d = kama_10[i] < kama_30[i]
        
        # KAMA slope
        kama_slope_long = False
        kama_slope_short = False
        if i >= 3 and not np.isnan(kama_10[i-3]):
            kama_slope_long = kama_10[i] > kama_10[i-3]
            kama_slope_short = kama_10[i] < kama_10[i-3]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_bullish = 35 < rsi[i] < 65
        rsi_bearish = 35 < rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # Donchian breakout signals
        donch_breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donch_breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # EMA alignment
        ema_aligned_long = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i] if not np.isnan(ema_200[i]) else ema_21[i] > ema_50[i]
        ema_aligned_short = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i] if not np.isnan(ema_200[i]) else ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple triggers for more trades) ===
        if bull_trend_1w:
            # Trigger 1: KAMA crossover with RSI confirmation
            if bull_trend_1d and rsi_bullish and vol_confirmed:
                new_signal = SIZE_BASE
            
            # Trigger 2: Pullback to EMA21 in uptrend
            elif price_near_ema21_long and ema_aligned_long and rsi[i] > 40:
                new_signal = SIZE_BASE
            
            # Trigger 3: Donchian breakout with volume
            elif donch_breakout_long and vol_ratio[i] > 1.2 and above_200:
                new_signal = SIZE_BASE
            
            # Trigger 4: RSI oversold bounce in bull regime
            elif rsi_oversold and bull_trend_1d and kama_slope_long:
                new_signal = SIZE_HALF
            
            # Trigger 5: KAMA slope turn with weekly support
            elif kama_slope_long and close[i] > hma_1w_aligned[i] * 0.95:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (multiple triggers for more trades) ===
        elif bear_trend_1w:
            # Trigger 1: KAMA crossover with RSI confirmation
            if bear_trend_1d and rsi_bearish and vol_confirmed:
                new_signal = -SIZE_BASE
            
            # Trigger 2: Bounce to EMA21 in downtrend
            elif price_near_ema21_short and ema_aligned_short and rsi[i] < 60:
                new_signal = -SIZE_BASE
            
            # Trigger 3: Donchian breakdown with volume
            elif donch_breakout_short and vol_ratio[i] > 1.2 and below_200:
                new_signal = -SIZE_BASE
            
            # Trigger 4: RSI overbought rejection in bear regime
            elif rsi_overbought and bear_trend_1d and kama_slope_short:
                new_signal = -SIZE_HALF
            
            # Trigger 5: KAMA slope turn with weekly resistance
            elif kama_slope_short and close[i] < hma_1w_aligned[i] * 1.05:
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