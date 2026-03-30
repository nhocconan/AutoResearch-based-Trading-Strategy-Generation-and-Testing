Looking at the experiment history, I see the **#1 killer is too_few_trades** (23 failed strategies!). The current Ichimoku+Alligator strategy is too restrictive with its 7+ stacked conditions, likely resulting in 0 trades.

The winning formula from the DB: **ONE strong signal + volume confirm + regime filter**. Let me try a cleaner approach.

**HYPOTHESIS**: Williams %R (simpler/faster than RSI) + Donchian breakout + volume spike. Williams %R < -70 signals momentum reversal up, combined with a 20-bar Donchian breakout for confirmation. This captures "oversold bounces" in both bull markets and bear rallies. HTF EMA(21) on 1d provides trend direction. Choppiness Index < 50 ensures we're not in a ranging market.

**Why it should work**: Williams %R oscillates 0 to -100, extremes (-80 to -100) catch reversal points. Donchian breakout confirms the move has momentum. Volume validates institutional participation. Choppiness avoids range-bound whipsaws. Simpler = fewer conditions = more trades = better statistics.

**Expected trades**: Williams %R < -70 and > -30 happens frequently (~30-40% of bars), combined with breakout + volume = ~75-200 trades/year on 4h. Target range.
#!/usr/bin/env python3
"""
Experiment #022: Williams %R + Donchian Breakout + Volume (4h)

HYPOTHESIS: Williams %R (<-70 = oversold bounce, >-30 = overbought dump) combined
with 20-bar Donchian breakout for momentum confirmation. Volume spike validates
institutional moves. HTF EMA(21) on 1d filters trend direction.
Choppiness < 50 avoids range-bound whipsaws.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Oversold bounce + breakout = strong long, pullback entries
- Bear: Overbought dump + breakdown = strong short, rally shorts
- Range: Choppiness filter reduces whipsaw losses
- Williams %R is faster than RSI, catches reversals earlier

TARGET: 100-250 total trades over 4 years (25-62/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_willr_donchian_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_willr(high, low, close, period=14):
    """Williams %R - momentum oscillator (-100 to 0)"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20-bar breakout structure"""
    n = len(high)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - trend vs range
    CHOP > 61.8 = ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx - 1]) if idx > 0 else high[idx] - low[idx], abs(low[idx] - close[idx - 1]) if idx > 0 else 0)
            sum_tr += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_sum = highest_high - lowest_low
        
        if range_sum > 0:
            chop[i] = 100 * (np.log(sum_tr) / np.log(range_sum)) if range_sum > 0 else np.nan
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF: EMA(21) on 1d for trend direction ===
    htf_ema = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    htf_price = df_1d['close'].values
    htf_bullish = htf_price > htf_ema
    htf_bearish = htf_price < htf_ema
    
    # Align HTF to 4h
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_willr(high, low, close, period=14)
    donchian_upper, _, donchian_lower = calculate_donchian(high, low, period=20)
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
    
    warmup = 60  # Enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # CHOP < 50 = trending, good for momentum trades
        trending = chop[i] < 50
        very_choppy = chop[i] > 62
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above 20-bar high = bullish momentum
        bullish_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Price breaks below 20-bar low = bearish momentum
        bearish_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === WILLIAMS %R MOMENTUM ===
        # <-70 = oversold (potential bounce)
        oversold = willr[i] < -70
        # >-30 = overbought (potential dump)
        overbought = willr[i] > -30
        # <-90 = extreme oversold (stronger signal)
        extreme_oversold = willr[i] < -90
        # >-10 = extreme overbought (stronger signal)
        extreme_overbought = willr[i] > -10
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG ENTRY: Williams %R oversold + bullish breakout + volume + trending
            # Relaxed: oversold OR extreme oversold (either works)
            # Add breakout for confirmation, volume for validation
            if (oversold or extreme_oversold) and bullish_breakout and vol_spike:
                if htf_bull or htf_bullish_aligned[i] > 0.25:  # Bull or neutral HTF
                    desired_signal = SIZE
            # Alternative: extreme oversold alone with breakout (catch the dip)
            elif extreme_oversold and bullish_breakout and not very_choppy:
                if htf_bull or htf_bullish_aligned[i] > 0.25:
                    desired_signal = SIZE
            
            # SHORT ENTRY: Williams %R overbought + bearish breakout + volume + trending
            if (overbought or extreme_overbought) and bearish_breakout and vol_spike:
                if htf_bear or htf_bearish_aligned[i] > 0.25:  # Bear or neutral HTF
                    desired_signal = -SIZE
            elif extreme_overbought and bearish_breakout and not very_choppy:
                if htf_bear or htf_bearish_aligned[i] > 0.25:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if Williams %R reaches overbought territory (reversal signal)
                if willr[i] > -20:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear and htf_bearish_aligned[i] > 0.75:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if Williams %R reaches oversold territory (reversal signal)
                if willr[i] < -80:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull and htf_bullish_aligned[i] > 0.75:
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