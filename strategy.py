#!/usr/bin/env python3
"""
Experiment #490: 4h Choppiness Regime + Daily HMA Bias + Fisher Transform Entries + ATR Stop
Hypothesis: 4h timeframe needs regime detection to avoid whipsaw in ranging markets.
Choppiness Index (CHOP) identifies trending vs ranging regimes. 
- Trending (CHOP < 38.2): Follow Daily HMA direction with momentum confirmation
- Ranging (CHOP > 61.8): Mean reversion with Fisher Transform extremes
- Neutral: Stay flat or reduce position size
Fisher Transform catches reversals better than RSI in bear/range markets (2025 test period).
Daily HMA provides HTF bias alignment. 2.5*ATR stoploss appropriate for 4h bars.
Conservative sizing (0.25) controls DD. Multiple entry paths ensure >=10 trades.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_daily_hma_fisher_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies trending vs ranging markets.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    Based on E.W. Dreiss formula.
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extreme values (-2 to +2 range typical).
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        highest = np.max(hl2[i-period+1:i+1])
        lowest = np.min(hl2[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        # Normalize price to 0-1 range
        x = (hl2[i] - lowest) / (highest - lowest)
        x = np.clip(x, 0.001, 0.999)  # Avoid log(0) or log(1)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]  # Previous value for crossover detection
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_momentum(close, period=10):
    """Calculate Rate of Change momentum."""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    rsi = calculate_rsi(close, 14)
    momentum = calculate_momentum(close, 10)
    
    # 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        regime_trending = chop[i] < 38.2
        regime_ranging = chop[i] > 61.8
        regime_neutral = not regime_trending and not regime_ranging
        
        # 4h HMA trend
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        hma_rising = hma_4h[i] > hma_4h[i-1] if i > 0 else False
        hma_falling = hma_4h[i] < hma_4h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher_signal[i] < -1.5 and fisher[i] >= -1.5 if not np.isnan(fisher_signal[i]) else False
        fisher_cross_down = fisher_signal[i] > 1.5 and fisher[i] <= 1.5 if not np.isnan(fisher_signal[i]) else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        mom_negative = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) - Follow Daily HMA direction ===
        
        if regime_trending:
            # Long: Daily bullish + 4h HMA bullish + momentum positive
            if daily_bullish and hma_4h_bullish and mom_positive:
                new_signal = SIZE_ENTRY
            
            # Short: Daily bearish + 4h HMA bearish + momentum negative
            elif daily_bearish and hma_4h_bearish and mom_negative:
                new_signal = -SIZE_ENTRY
            
            # Pullback entry in uptrend
            elif daily_bullish and hma_rising and rsi_pullback_long and fast_above_slow:
                new_signal = SIZE_ENTRY
            
            # Pullback entry in downtrend
            elif daily_bearish and hma_falling and rsi_pullback_short and fast_below_slow:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 61.8) - Mean reversion with Fisher ===
        
        elif regime_ranging:
            # Long: Fisher oversold cross + RSI oversold
            if fisher_cross_up and rsi_oversold:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher overbought cross + RSI overbought
            elif fisher_cross_down and rsi_overbought:
                new_signal = -SIZE_ENTRY
            
            # Extreme Fisher values (reversal play)
            elif fisher[i] < -2.0 and fisher[i] > fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                new_signal = SIZE_ENTRY
            
            elif fisher[i] > 2.0 and fisher[i] < fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME - Reduced size or flat ===
        
        else:
            # Only take high-confidence trades with reduced size
            if daily_bullish and fisher_cross_up and rsi[i] < 50:
                new_signal = SIZE_HALF
            elif daily_bearish and fisher_cross_down and rsi[i] > 50:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals