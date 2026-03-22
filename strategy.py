#!/usr/bin/env python3
"""
Experiment #020: 30m Adaptive KAMA Trend with 4h HMA Regime + Fisher Transform Entries

Hypothesis: KAMA adapts to market volatility better than EMA, reducing whipsaws in 
range markets while catching trends. Combined with 4h HMA for regime filter and 
Fisher Transform for precise entry timing, this should work in both bull and bear 
markets. Fisher Transform excels at catching reversals where RSI fails.

Key innovations vs failed experiments:
1. KAMA instead of EMA/HMA on LTF - adapts to volatility
2. Fisher Transform instead of RSI - better reversal detection  
3. Volume confirmation using taker_buy_volume - filters false breakouts
4. BB Width squeeze detection - catches explosive moves
5. LOOSENED entry conditions - MUST generate 10+ trades per symbol

Timeframe: 30m (REQUIRED for exp#020)
HTF: 4h HMA for trend regime
Position sizing: 0.20-0.30 discrete, 2.5*ATR stoploss
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_fisher_4h_hma_vol_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0] = 0
    
    volatility = np.zeros(n)
    for i in range(1, n):
        vol_sum = 0
        for j in range(1, min(period, i+1)):
            vol_sum += np.abs(close[i-j+1] - close[i-j])
        volatility[i] = vol_sum if vol_sum > 0 else 1e-10
    
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    er[0] = 0
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range, excellent for reversal detection.
    Uses HL2 for better signal quality.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    if n < period:
        return fisher, fisher_signal
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            fisher[i] = 0
            continue
        
        # Normalize
        normalized = (hl2 - lowest) / (highest - lowest)
        normalized = 0.999 * (2 * normalized - 1)  # Scale to -0.999 to +0.999
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_bb_width(close, period=20, std_mult=2.0):
    """
    Bollinger Band Width - measures volatility squeeze.
    """
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / (sma + 1e-10)
    
    return width, sma, upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR for stoploss."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for HTF trend."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return wma3.values

def calculate_volume_ratio(taker_buy_volume, volume, period=14):
    """Calculate volume ratio for confirmation."""
    ratio = taker_buy_volume / (volume + 1e-10)
    avg_ratio = pd.Series(ratio).rolling(window=period, min_periods=period).mean().values
    return ratio, avg_ratio

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend regime
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama = calculate_kama(close, period=10)
    fisher, fisher_signal = calculate_fisher(high, low, close, period=9)
    bb_width, bb_mid, bb_upper, bb_lower = calculate_bb_width(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    vol_ratio, vol_avg = calculate_volume_ratio(taker_buy_vol, volume, 14)
    
    # EMA for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0 or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend regime (HTF)
        bull_regime_4h = close[i] > hma_4h_aligned[i]
        bear_regime_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend via KAMA
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # EMA trend confirmation
        ema_bull = not np.isnan(ema_50[i]) and ema_21[i] > ema_50[i]
        ema_bear = not np.isnan(ema_50[i]) and ema_21[i] < ema_50[i]
        
        # Fisher Transform signals (reversal detection) - LOOSENED thresholds
        fisher_bull = False
        fisher_bear = False
        
        if i >= 2 and not np.isnan(fisher[i-1]) and not np.isnan(fisher[i]):
            # Bullish: Fisher crosses above -1.0 from below (loosened from -1.5)
            fisher_bull = fisher[i] > -1.0 and fisher[i-1] <= -1.0
            # Bearish: Fisher crosses below +1.0 from above (loosened from +1.5)
            fisher_bear = fisher[i] < 1.0 and fisher[i-1] >= 1.0
        
        # Volume confirmation
        vol_confirm_long = vol_ratio[i] > 0.40  # Some buying pressure
        vol_confirm_short = vol_ratio[i] < 0.60  # Some selling pressure
        
        # BB Width squeeze (volatility expansion)
        bb_squeeze = False
        if i > 30 and not np.isnan(bb_width[i]):
            recent_width = bb_width[max(0,i-30):i+1]
            valid_width = recent_width[~np.isnan(recent_width)]
            if len(valid_width) > 10:
                bb_squeeze = bb_width[i] < np.percentile(valid_width, 35)
        
        # Price action
        price_above_kama = close[i] > kama[i] * 1.0005
        price_below_kama = close[i] < kama[i] * 0.9995
        
        # KAMA slope
        kama_rising = False
        kama_falling = False
        if i >= 3 and not np.isnan(kama[i-1]) and not np.isnan(kama[i-3]):
            kama_rising = kama[i] > kama[i-3]
            kama_falling = kama[i] < kama[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        if bull_regime_4h:  # Only long in 4h bull regime
            # Primary: KAMA cross + Fisher bullish + volume
            if price_above_kama and fisher_bull and vol_confirm_long:
                new_signal = SIZE_BASE
            
            # Secondary: EMA bull + Fisher reversal + BB squeeze breakout
            elif ema_bull and fisher_bull and close[i] > bb_upper[i] * 0.998:
                new_signal = SIZE_BASE
            
            # Tertiary: Simple KAMA cross in bull regime with rising KAMA
            elif price_above_kama and kama_rising and vol_ratio[i] > 0.45:
                new_signal = SIZE_HALF
            
            # Quaternary: KAMA bull + volume confirmation (most permissive)
            elif kama_bull and vol_confirm_long and ema_bull:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        elif bear_regime_4h:  # Only short in 4h bear regime
            # Primary: KAMA cross + Fisher bearish + volume
            if price_below_kama and fisher_bear and vol_confirm_short:
                new_signal = -SIZE_BASE
            
            # Secondary: EMA bear + Fisher reversal + BB squeeze breakdown
            elif ema_bear and fisher_bear and close[i] < bb_lower[i] * 1.002:
                new_signal = -SIZE_BASE
            
            # Tertiary: Simple KAMA cross in bear regime with falling KAMA
            elif price_below_kama and kama_falling and vol_ratio[i] < 0.55:
                new_signal = -SIZE_HALF
            
            # Quaternary: KAMA bear + volume confirmation (most permissive)
            elif kama_bear and vol_confirm_short and ema_bear:
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