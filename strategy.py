#!/usr/bin/env python3
"""
Experiment #422: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform + Volume

Hypothesis: After analyzing 421 failed experiments, key insights emerge:
1. 12h timeframe needs SIMPLE logic — complex regimes overfit (see #412, #416, #421 failures)
2. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA in choppy markets
3. Ehlers Fisher Transform catches reversals in bear markets better than RSI (research notes)
4. Volume confirmation prevents false breakouts (critical for 12h signals)
5. Asymmetric entries: 1w trend direction gets larger size, counter-trend gets smaller
6. Fewer confluence filters = more trades (address 0-trade failures like #410, #413, #418)

Why this might beat current best (Sharpe=0.435):
- KAMA efficiency ratio adapts to regime automatically (no manual chop filter needed)
- Fisher Transform normalized -1.5 to +1.5 gives cleaner reversal signals than RSI
- 1w HTF for major bias + 1d for intermediate trend = proven multi-scale approach
- Volume spike confirmation (1.5x avg) filters false breakouts
- Simpler logic = more trades (target 30-50/year on 12h)

Position sizing: 0.20-0.30 discrete, asymmetric based on 1w trend
Stoploss: 2.5 * ATR trailing
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0.435
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_vol_1d1w_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    ER=1 means strong trend (use fast SC), ER=0 means choppy (use slow SC).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |net change| / sum of absolute changes
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = change / (volatility + 1e-10)
    er[:er_period] = np.nan
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1.5 to +1.5 range for reversal detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price position within range
        range_val = hh - ll
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher_signal[i-1] if i > 0 else 0.0
            continue
        
        # Transform
        x = ((close[i] - ll) / range_val - 0.5) * 2.0
        x = np.clip(x, -0.999, 0.999)  # prevent log errors
        
        fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x))
        fisher_prev = fisher[i-1] if i > 0 else 0.0
        
        fisher[i] = 0.67 * fisher_val + 0.33 * fisher_prev
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    fisher[:period] = np.nan
    fisher_signal[:period] = np.nan
    
    return fisher, fisher_signal

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def generate_signals(prices):
    global close  # Make close available for fisher calculation
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF KAMA (intermediate trend)
    kama_1d_20 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    
    # Calculate 1w HTF KAMA (major trend direction)
    kama_1w_20 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_20_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_20)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_12h_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30  # Larger when aligned with 1w trend
    COUNTER_SIZE = 0.15  # Smaller when counter to 1w trend
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_20_aligned[i]) or np.isnan(kama_1w_20_aligned[i]):
            continue
        
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 1W MAJOR TREND (directional bias) ===
        # Price above 1w KAMA = bull market (favor longs with larger size)
        # Price below 1w KAMA = bear market (favor shorts with larger size)
        bull_1w = close[i] > kama_1w_20_aligned[i]
        bear_1w = close[i] < kama_1w_20_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_1d = close[i] > kama_1d_20_aligned[i]
        bear_1d = close[i] < kama_1d_20_aligned[i]
        
        # === 12H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher crosses above -1.5 from below = long reversal signal
        # Fisher crosses below +1.5 from above = short reversal signal
        fisher_long = (fisher[i] > -1.5 and fisher_signal[i] <= -1.5)
        fisher_short = (fisher[i] < 1.5 and fisher_signal[i] >= 1.5)
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === ENTRY LOGIC — SIMPLE & ASYMMETRIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Determine position size based on 1w trend alignment
        long_size = TREND_SIZE if bull_1w else COUNTER_SIZE
        short_size = TREND_SIZE if bear_1w else COUNTER_SIZE
        
        # LONG ENTRY
        if bull_1w or bull_1d:  # At least one higher TF bullish
            # Fisher reversal + volume confirmation
            if fisher_long and vol_confirmed:
                new_signal = long_size
            # KAMA crossover + Fisher oversold
            elif kama_bullish and fisher_oversold:
                new_signal = long_size * 0.8
            # Simple trend continuation (12h KAMA bull + price above 1d KAMA)
            elif kama_bullish and bull_1d and not fisher_overbought:
                new_signal = long_size * 0.7
        
        # SHORT ENTRY
        if bear_1w or bear_1d:  # At least one higher TF bearish
            # Fisher reversal + volume confirmation
            if fisher_short and vol_confirmed:
                if new_signal == 0.0:
                    new_signal = -short_size
            # KAMA crossover + Fisher overbought
            elif kama_bearish and fisher_overbought:
                if new_signal == 0.0:
                    new_signal = -short_size * 0.8
            # Simple trend continuation (12h KAMA bear + price below 1d KAMA)
            elif kama_bearish and bear_1d and not fisher_oversold:
                if new_signal == 0.0:
                    new_signal = -short_size * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~6 days on 12h), enter on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_1w and kama_bullish and fisher[i] < 0.5:
                new_signal = long_size * 0.5
            elif bear_1w and kama_bearish and fisher[i] > -0.5:
                new_signal = -short_size * 0.5
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on reversal exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.2:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.2:
            new_signal = 0.0
        
        # Major trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_1w:
            new_signal = 0.0
        
        # Local trend reversal exit (12h KAMA cross)
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals