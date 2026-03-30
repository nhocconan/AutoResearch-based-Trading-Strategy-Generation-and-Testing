#!/usr/bin/env python3
"""
Experiment #022: TRIX Momentum + Donchian Breakout + Volume Confirmation (12h)

HYPOTHESIS: Combine TRIX momentum oscillator with Donchian breakout for robust 12h entries:
1. TRIX(12) - triple EMA momentum, filters noise, catches trend starts/ends
2. Donchian(20) breakout - price channel structure, proven edge
3. Volume spike - trade validation
4. Choppiness regime filter - avoid ranging markets
5. 1d HTF for trend direction filter

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: TRIX crosses above 0 + price breaks above Donchian high + vol spike = strong long
- Bear: TRIX crosses below 0 + price breaks below Donchian low + vol spike = strong short
- Range (CHOP > 61): Skip trades, reduces whipsaws
- TRIX is explicitly mentioned as untried/novel in DB, with ETHUSDT achieving 1.32 Sharpe

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=12):
    """
    TRIX - Triple Exponential Average
    TRIX = Rate of change of triple EMA
    Positive TRIX = upward momentum, negative = downward
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX as percentage change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period, n):
        if ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    return trix

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - price channel breakout
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        start = i - period + 1
        upper[i] = np.max(high[start:i+1])
        lower[i] = np.min(low[start:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Sum of log range over period
        log_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if high[j] > low[j] and close[j] > 0:
                log_sum += np.log(high[j] / low[j])
        
        # Log of total range
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        
        if period_high > period_low and period_low > 0:
            log_range = np.log(period_high / period_low)
            if log_range > 0:
                chop[i] = 100 * log_sum / (period * np.log(log_range))
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d HMA for trend direction ===
    hma_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21).mean().values
    htf_trend = df_1d['close'].values > hma_1d  # 1d price above HMA = bull
    htf_trend_aligned = align_htf_to_ltf(prices, df_1d, htf_trend.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # TRIX momentum
    trix = calculate_trix(close, period=12)
    
    # Donchian channel
    donchian_upper, donchian_lower, _ = calculate_donchian(high, low, period=20)
    
    # Choppiness index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 60  # TRIX needs ~36, Donchian needs 20, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Choppiness > 61.8 = ranging, skip trades
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull = htf_trend_aligned[i] > 0.5 if not np.isnan(htf_trend_aligned[i]) else False
        
        # === TRIX CROSS ===
        trix_above_zero = trix[i] > 0
        trix_below_zero = trix[i] < 0
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0 if i > 0 else False
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0 if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        price_breaks_high = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        price_breaks_low = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Skip in ranging markets
            if is_ranging:
                pass
            else:
                # LONG: TRIX crosses up + price breaks above Donchian high + vol spike
                # Or: TRIX already positive + price above Donchian high + strong vol + trending
                long_cond1 = trix_cross_up and price_breaks_high and vol_spike
                long_cond2 = trix_above_zero and price_breaks_high and vol_spike and is_trending
                
                if (long_cond1 or long_cond2) and htf_bull:
                    desired_signal = SIZE
                
                # SHORT: TRIX crosses down + price breaks below Donchian low + vol spike
                # Or: TRIX already negative + price below Donchian low + strong vol + trending
                short_cond1 = trix_cross_down and price_breaks_low and vol_spike
                short_cond2 = trix_below_zero and price_breaks_low and vol_spike and is_trending
                
                if short_cond1 or short_cond2:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns negative
                if trix[i] < 0:
                    desired_signal = 0.0
                
                # Exit if ranging
                if is_ranging:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns positive
                if trix[i] > 0:
                    desired_signal = 0.0
                
                # Exit if ranging
                if is_ranging:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
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