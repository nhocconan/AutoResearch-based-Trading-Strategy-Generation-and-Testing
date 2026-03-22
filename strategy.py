#!/usr/bin/env python3
"""
Experiment #012: 1d Mean Reversion with 1W HMA Trend + Multi-Signal Confluence

Hypothesis: After analyzing 11 failed experiments, the pattern is clear:
1. Daily timeframe naturally reduces trade frequency → less fee drag
2. Mean reversion at extremes works better than trend-following in bear/range markets
3. Multi-signal confluence reduces false entries while maintaining trade count
4. 1W HMA provides ultra-stable trend bias without whipsaw

This 1d strategy combines:

1. 1W HMA trend bias: Only long if price > 1w_HMA, only short if price < 1w_HMA.
   Weekly HMA is extremely stable, filters out noise from daily fluctuations.

2. RSI(14) extremes: Long when RSI < 35, Short when RSI > 65.
   Less extreme than CRSI to ensure sufficient trade count on 1d.

3. Bollinger Band %B: Long when %B < 0.15, Short when %B > 0.85.
   Confirms price at band extremes for mean reversion entry.

4. Volume confirmation: Entry volume > 1.3 * 20d avg volume.
   Ensures institutional participation at extremes.

5. Z-score filter: |z-score(20)| > 1.8 for entry confirmation.
   Statistical extreme confirmation.

6. ATR(14) trailing stop: 2.5*ATR to protect from crashes.

7. Regime-adaptive sizing: 0.25 base, 0.30 with full confluence.

Why this should beat #002 (Sharpe=0.123):
- 1d timeframe = fewer trades, less fee drag (target 30-50 trades/year)
- Multi-signal confluence = higher win rate on each trade
- Mean reversion works in both bull and bear markets
- 1W HMA filter prevents catching falling knives in crashes
- Volume confirmation ensures real institutional moves

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_bb_zscore_1w_hma_vol_atr_v1"
timeframe = "1d"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and %B indicator."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # %B = (price - lower) / (upper - lower)
    bb_width = upper - lower
    pct_b = (close_s - lower) / bb_width.replace(0, np.inf)
    
    return sma.values, upper.values, lower.values, pct_b.values

def calculate_zscore(close, lookback=20):
    """Calculate rolling z-score of price."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = close_s.rolling(window=lookback, min_periods=lookback).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def calculate_volume_spike(volume, lookback=20, threshold=1.3):
    """Detect volume spikes above threshold * rolling average."""
    volume_s = pd.Series(volume)
    vol_avg = volume_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume_s > (threshold * vol_avg)
    return vol_spike.values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_20, bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, 20, 2.0)
    zscore_20 = calculate_zscore(close, 20)
    vol_spike = calculate_volume_spike(volume, 20, 1.3)
    
    # Calculate SMA200 for additional trend filter
    close_s = pd.Series(close)
    sma_200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(250, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(pct_b[i]) or np.isnan(zscore_20[i]):
            continue
        
        # === 1W HMA TREND BIAS (Ultra-stable HTF filter) ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND %B EXTREMES ===
        bb_oversold = pct_b[i] < 0.15
        bb_overbought = pct_b[i] > 0.85
        
        # === Z-SCORE EXTREMES ===
        zscore_extreme_long = zscore_20[i] < -1.8
        zscore_extreme_short = zscore_20[i] > 1.8
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: Need HTF bullish bias + at least 2 of 3 mean reversion signals
        if bull_bias:
            mr_signals_long = sum([rsi_oversold, bb_oversold, zscore_extreme_long])
            
            if mr_signals_long >= 2:
                if volume_confirmed:
                    new_signal = CONFIRMED_SIZE
                    signal_strength = 2
                else:
                    new_signal = BASE_SIZE
                    signal_strength = 1
        
        # SHORT ENTRY: Need HTF bearish bias + at least 2 of 3 mean reversion signals
        elif bear_bias:
            mr_signals_short = sum([rsi_overbought, bb_overbought, zscore_extreme_short])
            
            if mr_signals_short >= 2:
                if volume_confirmed:
                    new_signal = -CONFIRMED_SIZE
                    signal_strength = 2
                else:
                    new_signal = -BASE_SIZE
                    signal_strength = 1
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish
            if position_side > 0 and bear_bias:
                trend_exit = True
            # Exit short if trend turns bullish
            if position_side < 0 and bull_bias:
                trend_exit = True
        
        # === MEAN REVERSION EXIT (opposite extreme) ===
        mr_exit = False
        if in_position and position_side != 0:
            # Exit long when overbought
            if position_side > 0 and (rsi_14[i] > 70 or pct_b[i] > 0.85):
                mr_exit = True
            # Exit short when oversold
            if position_side < 0 and (rsi_14[i] < 30 or pct_b[i] < 0.15):
                mr_exit = True
        
        # Apply stoploss or exit conditions
        if stoploss_triggered or trend_exit or mr_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals