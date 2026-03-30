#!/usr/bin/env python3
"""
Experiment #022: ATR Channel Breakout + Williams %R + 12h Trend (4h)

HYPOTHESIS: ATR channels provide volatility-adjusted breakout signals.
Williams %R is simpler than RSI for catching reversals.
12h EMA confirms trend direction. Volume spike confirms conviction.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above ATR channel upper band + Williams %R rising + vol spike + 12h bull
- Bear: Price breaks below ATR channel lower band + Williams %R falling + vol spike + 12h bear
- ATR channels expand during volatility (crashes/rallies) = self-adjusting

KEY INSIGHT from failures: Current strategy uses 5 AND conditions = 0 trades.
NEW approach: ATR channel breakout = main signal. Williams %R = confirmation (optional).
Volume spike = required. 12h trend = required. This is 2-3 conditions per entry.

TARGET: 75-150 total trades over 4 years (19-37/year) - safe zone.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atr_channel_willr_vol_12h_v1"
timeframe = "4h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_ = highest_high - lowest_low
        
        if range_ > 0:
            willr[i] = -100 * (highest_high - close[i]) / range_
        else:
            willr[i] = -50  # Default when range is zero
    
    return willr

def calculate_atr_channel(high, low, close, period=20, atr_mult=2.5):
    """
    ATR Channel Breakout
    - Upper band: EMA + ATR_mult * ATR
    - Lower band: EMA - ATR_mult * ATR
    - Middle: EMA
    """
    n = len(close)
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period=14)
    
    upper_band = ema + atr_mult * atr
    lower_band = ema - atr_mult * atr
    
    return ema, upper_band, lower_band

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
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
    
    dx = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 0:
            di_plus = 100 * plus_dm_smooth[i] / atr[i]
            di_minus = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus + di_minus
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus - di_minus) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - trend vs range
    < 38.2 = trending (use trend-following)
    > 61.8 = choppy (avoid or use mean-reversion)
    """
    n = len(close)
    chop = np.full(n, 50.0)  # Default neutral
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # HTF 12h EMA for trend
    htf_ema = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    htf_ema_aligned = align_htf_to_ltf(prices, df_12h, htf_ema)
    
    # HTF ADX for regime
    htf_adx = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    htf_adx_aligned = align_htf_to_ltf(prices, df_12h, htf_adx)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_20, upper_band, lower_band = calculate_atr_channel(high, low, close, period=20, atr_mult=2.5)
    willr = calculate_williams_r(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = SIZE / 2
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    highest_after_entry = 0.0
    lowest_after_entry = 0.0
    
    warmup = 100  # ATR channel needs ~20, williams %R needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(htf_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (required filter) ===
        htf_trend_bull = close[i] > htf_ema_aligned[i] if not np.isnan(htf_ema_aligned[i]) else False
        htf_trend_bear = close[i] < htf_ema_aligned[i] if not np.isnan(htf_ema_aligned[i]) else False
        htf_strong_trend = htf_adx_aligned[i] > 25 if not np.isnan(htf_adx_aligned[i]) else False
        
        # === LOCAL TREND CONDITIONS ===
        local_trending = adx[i] > 20 if not np.isnan(adx[i]) else False
        not_choppy = chop[i] < 61.8  # Not too choppy
        
        # === VOLUME CONFIRMATION (required) ===
        vol_confirm = vol_ratio[i] > 1.4
        
        # === WILLIAMS % R MOMENTUM ===
        willr_oversold = willr[i] < -80  # Very oversold
        willr_overbought = willr[i] > -20  # Very overbought (willr is negative)
        willr_rising = willr[i] > willr[i-1] if i > 0 and not np.isnan(willr[i-1]) else False
        willr_falling = willr[i] < willr[i-1] if i > 0 and not np.isnan(willr[i-1]) else False
        
        # === ATR CHANNEL SIGNALS ===
        # Breakout: price closes outside bands
        above_upper = close[i] > upper_band[i]
        below_lower = close[i] < lower_band[i]
        
        # Retracement entry: price returns to middle from extreme
        near_lower_retreat = close[i] > lower_band[i] and close[i-1] <= lower_band[i-1] if i > 0 else False
        near_upper_retreat = close[i] < upper_band[i] and close[i-1] >= upper_band[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG ENTRY: Price breaks above upper band OR returns from lower band
            # Requires: HTF bull, vol spike, not too choppy
            long_breakout = above_upper and vol_confirm
            long_retrace = near_lower_retreat and willr_oversold and vol_confirm
            
            if (long_breakout or long_retrace) and not_choppy:
                if htf_trend_bull or htf_strong_trend:  # HTF confirms or neutral
                    desired_signal = SIZE
            
            # SHORT ENTRY: Price breaks below lower band OR returns from upper band
            # Requires: HTF bear, vol spike, not too choppy
            short_breakout = below_lower and vol_confirm
            short_retrace = near_upper_retreat and willr_overbought and vol_confirm
            
            if (short_breakout or short_retrace) and not_choppy:
                if htf_trend_bear or htf_strong_trend:  # HTF confirms or neutral
                    desired_signal = -SIZE
        
        # === EXIT / STOP LOSS ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls back below ATR channel middle
                if close[i] < ema_20[i]:
                    desired_signal = 0.0
                
                # Exit if Williams %R turns bearish
                if willr_falling and willr[i] < -30:
                    # Take partial profit
                    if close[i] > entry_price * 1.02:  # 2% profit
                        desired_signal = HALF_SIZE
                
                # Exit if HTF turns bearish
                if htf_trend_bear and not htf_strong_trend:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises back above ATR channel middle
                if close[i] > ema_20[i]:
                    desired_signal = 0.0
                
                # Exit if Williams %R turns bullish
                if willr_rising and willr[i] > -70:
                    # Take partial profit
                    if close[i] < entry_price * 0.98:  # 2% profit
                        desired_signal = -HALF_SIZE
                
                # Exit if HTF turns bullish
                if htf_trend_bull and not htf_strong_trend:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
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