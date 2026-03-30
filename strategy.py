#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian Breakout + 1d Trend Filter

HYPOTHESIS: Simple price channel breakout works across bull/bear/range:
- Long: 12h price breaks Donchian high(20) + 1d trend bull + volume confirmation
- Short: 12h price breaks Donchian low(20) + 1d trend bear + volume confirmation
- Exit: Price reverts to Donchian middle OR 2.5*ATR stoploss

WHY IT WORKS:
- Donchian(20) on 12h = 10-day breakout, captures multi-day trends
- 1d SMA(50) for trend direction = proven HTF filter
- Volume confirmation = prevents false breakouts
- Simple 3-condition entry = 75-150 trades over 4 years (low fee drag)
- Mirrors proven DB winner (Sharpe 1.49 on SOL)

TARGET: 75-150 total trades over 4 years (19-37/year)
TIMEFRAME: 12h
LEVERAGE: 1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_trend_vol_v1"
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
    """Donchian Channel - price channel breakout indicator"""
    n = len(high)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """ADX for trend strength - ADX > 20 = trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
                adx[i] = dx
    
    adx_smooth = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    htf_price = df_1d['close'].values
    
    # HTF trend: price above SMA = bull, below = bear
    htf_bull = htf_price > sma_1d
    htf_bear = htf_price < sma_1d
    
    # Align HTF to LTF
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_high, donchian_mid, donchian_low = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Detect breakout ===
    # Breakout: price exceeds prior Donchian high with volume
    breakout_high = close > donchian_high
    breakout_low = close < donchian_low
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian 20 + ATR 14 + volume 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === GET HTF TREND ===
        htf_is_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_is_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === CONDITIONS ===
        vol_spike = vol_ratio[i] > 1.4  # Volume > 1.4x average
        trending = adx[i] > 18  # ADX > 18 = trending
        
        # Price near channel edge (potential breakout)
        near_high = close[i] >= donchian_high[i] * 0.98
        near_low = close[i] <= donchian_low[i] * 1.02
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above Donchian high + HTF bull + volume + trending
            if breakout_high[i] and htf_is_bull and vol_spike and trending:
                desired_signal = SIZE
            # Alternative: Near breakout with strong confirmation
            elif near_high and htf_is_bull and vol_spike and trending and adx[i] > 22:
                desired_signal = SIZE
            
            # SHORT: Breakout below Donchian low + HTF bear + volume + trending
            if breakout_low[i] and htf_is_bear and vol_spike and trending:
                desired_signal = -SIZE
            elif near_low and htf_is_bear and vol_spike and trending and adx[i] > 22:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_is_bear:
                    desired_signal = 0.0
                
                # Exit if price falls back below middle channel
                if close[i] < donchian_mid[i] and adx[i] < 15:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_is_bull:
                    desired_signal = 0.0
                
                # Exit if price rises back above middle channel
                if close[i] > donchian_mid[i] and adx[i] < 15:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
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