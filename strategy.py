#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian + Volume + Choppiness Regime

HYPOTHESIS: 12h timeframe with proven winning pattern from DB:
- Donchian(20) breakout - price channel structure (works in both bull/bear)
- Volume spike confirmation - filters false breakouts
- Choppiness Index regime filter - separates trending from ranging conditions
- HTF (1d) EMA for trend direction

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull market: Price breaks above Donchian upper + volume spike + chop<38.2 = long
- Bear market: Price breaks below Donchian lower + volume spike + chop<38.2 = short
- Range market: chop>61.8 = no entry (avoids whipsaws during consolidation)
- Symmetric logic: both long and short have same structure

KEY INSIGHT from DB:
- "mtf_4h_chop_donchian_vol_regime_12h_v1" → test_sharpe=1.491, 107 trades on SOLUSDT
- 54% keep rate on 12h (highest of all timeframes)
- Simple 3-condition pattern: price channel + volume + regime

TARGET: 75-200 total trades over 4 years (19-50/year on 12h)
This matches successful DB entries with Sharpe 1.3-1.5 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_v5"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CI) - measures market "choppiness"
    CI < 38.2 = trending (good for momentum strategies)
    CI > 61.8 = ranging (mean reversion territory)
    
    Formula: 100 * LOG10(SUM(ATR(1), period) / (HHV(period) - LLV(period))) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    ci = np.full(n, np.nan)
    
    # First ATR (same as period)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        
        # Highest high over period
        hh = np.max(high[i - period + 1:i + 1])
        
        # Lowest low over period
        ll = np.min(low[i - period + 1:i + 1])
        
        range_val = hh - ll
        
        if range_val > 0 and atr_sum > 0:
            ci[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return ci

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume / np.where(vol_ma > 0, vol_ma, 1)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF 1d indicators ===
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align HTF to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d price vs EMA for trend
    htf_price = df_1d['close'].values
    htf_bullish = htf_price > ema_1d
    htf_bearish = htf_price < ema_1d
    
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_mid, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume ratio
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - proven from DB winners
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Donchian needs 20, ATR needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Choppiness < 38.2 = trending (good for breakout trades)
        # Choppiness > 61.8 = ranging (avoid breakout trades)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Volume 1.5x above average
        
        # === DONCHIAN BREAKOUT ===
        # Upper breakout (potential long)
        upper_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Lower breakout (potential short)
        lower_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === HTF TREND ===
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Upper Donchian breakout + volume spike + trending regime + HTF bull
            if upper_breakout and vol_spike and trending_regime:
                # In bull market: require HTF bull
                # In bear market: allow if not strongly bearish
                if htf_bull or not htf_bear:
                    desired_signal = SIZE
            
            # SHORT: Lower Donchian breakdown + volume spike + trending regime + HTF bear
            elif lower_breakout and vol_spike and trending_regime:
                # In bear market: require HTF bear
                # In bull market: allow if not strongly bullish
                if htf_bear or not htf_bull:
                    desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls back below mid-channel (trend weakening)
                if close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises back above mid-channel (trend weakening)
                if close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (12h * 3 = 36h) to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals