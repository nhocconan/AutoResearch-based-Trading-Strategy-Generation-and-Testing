#!/usr/bin/env python3
"""
Experiment #508: 4h Volatility Regime Adaptive with Daily HMA Bias

Hypothesis: After analyzing 496+ failed experiments, the key insight is that 4h 
timeframe needs VOLATILITY-BASED regime adaptation, not complex asymmetric logic.
High volatility = mean reversion (panic/reversal trades). Low volatility = trend 
following (breakout trades). This adapts to both bull (2021) and bear (2022, 2025) 
markets.

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Bull: price > 1d HMA (favor long entries)
   - Bear: price < 1d HMA (favor short entries)
   - Simple binary filter, not complex asymmetric logic

2. VOLATILITY REGIME (ATR ratio):
   - ATR(7)/ATR(30) > 1.8 = HIGH VOL (mean reversion mode)
   - ATR(7)/ATR(30) < 1.2 = LOW VOL (trend following mode)
   - Between = neutral (reduced position size)

3. ADAPTIVE ENTRY LOGIC:
   - HIGH VOL + Bull: RSI(7) < 25 long (panic buy)
   - HIGH VOL + Bear: RSI(7) > 75 short (panic short)
   - LOW VOL + Bull: Price > Donchian(20) high long (breakout)
   - LOW VOL + Bear: Price < Donchian(20) low short (breakdown)

4. BOLLINGER BAND CONFIRMATION:
   - Mean reversion: price must touch BB(20, 2.5) bands
   - Trend: price must close outside BB(20, 2.0)

5. ATR(14) TRAILING STOP at 2.5x:
   - Tighter stop for 4h timeframe
   - Signal → 0 when price moves 2.5*ATR against position

6. POSITION SIZING: 0.25 discrete (conservative for 4h volatility)
   - High vol: 0.25 (panic trades have higher win rate)
   - Low vol: 0.20 (breakouts have more false signals)

Why this should work on 4h:
- Volatility regime adapts to market conditions (panic vs grind)
- Daily HMA provides robust trend bias without whipsaw
- Looser RSI thresholds (25/75) ensure sufficient trades
- Should generate 30-60 trades/year per symbol
- Works in both bull (breakout) and bear (panic reversal) markets

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_regime_daily_hma_rsi_donchian_bb_atr_v1"
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper_25, bb_lower_25, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    bb_upper_20, bb_lower_20, _ = calculate_bollinger_bands(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volatility ratio (ATR7 / ATR30)
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_HIGH_VOL = 0.25  # Panic trades have higher win rate
    SIZE_LOW_VOL = 0.20   # Breakouts have more false signals
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper_25[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        high_vol = vol_ratio[i] > 1.8
        low_vol = vol_ratio[i] < 1.2
        # neutral: between 1.2 and 1.8
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_LOW_VOL  # default
        
        # HIGH VOLATILITY: Mean Reversion (panic trades)
        if high_vol:
            current_size = SIZE_HIGH_VOL
            if bull_regime:
                # Bull + High Vol: Buy panic dips
                if rsi_7[i] < 25 and close[i] <= bb_lower_25[i]:
                    new_signal = current_size
            elif bear_regime:
                # Bear + High Vol: Short panic rallies
                if rsi_7[i] > 75 and close[i] >= bb_upper_25[i]:
                    new_signal = -current_size
        
        # LOW VOLATILITY: Trend Following (breakouts)
        elif low_vol:
            current_size = SIZE_LOW_VOL
            if bull_regime:
                # Bull + Low Vol: Breakout long
                if close[i] > donchian_upper[i-1] and close[i] > bb_upper_20[i]:
                    new_signal = current_size
            elif bear_regime:
                # Bear + Low Vol: Breakdown short
                if close[i] < donchian_lower[i-1] and close[i] < bb_lower_20[i]:
                    new_signal = -current_size
        
        # NEUTRAL VOLATILITY: Reduced activity
        else:
            # Only take strongest signals in neutral vol
            if bull_regime and rsi_7[i] < 20:
                new_signal = SIZE_LOW_VOL * 0.5
            elif bear_regime and rsi_7[i] > 80:
                new_signal = -SIZE_LOW_VOL * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals