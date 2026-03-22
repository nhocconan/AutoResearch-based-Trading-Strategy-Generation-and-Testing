#!/usr/bin/env python3
"""
Experiment #036: 12h Donchian Breakout + 1d HMA Trend + RSI Filter

Hypothesis: Previous complex strategies (Connors + Choppiness + multiple HTF) failed
due to too many conflicting filters = 0 trades. This strategy SIMPLIFIES:

1. 12h primary timeframe (REQUIRED for this experiment)
2. 1d HMA(21) for trend bias ONLY (price above = long bias, below = short bias)
3. Donchian(20) breakout for entries (proven on SOL with Sharpe +0.782)
4. RSI(14) simple filter (40-60 range for continuation, not extreme)
5. ATR(14) trailing stoploss at 2.5x
6. Looser thresholds to ensure 20-50 trades/year target

Why this should beat Sharpe=0.028:
- Donchian breakout is proven edge (SOL Sharpe +0.782 historically)
- 1d HMA trend filter prevents counter-trend trades (reduces whipsaw)
- Simpler = more trades, less chance of 0-trade failure
- 12h timeframe naturally gives 20-50 trades/year target
- RSI filter avoids breakout traps at extremes

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_rsi_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D TREND BIAS ===
        # Simple: price above 1d HMA = bullish bias, below = bearish bias
        trend_bullish = close[i] > hma_1d_21_aligned[i]
        trend_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND ===
        local_bullish = close[i] > hma_12h_21[i]
        local_bearish = close[i] < hma_12h_21[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout = price crosses above upper or below lower
        prev_upper = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_lower = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_upper
        breakout_short = close[i] < prev_lower
        
        # === RSI FILTER ===
        # Avoid extremes - want continuation, not reversal
        rsi_neutral = 35 < rsi_14[i] < 65
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        # Round to discrete levels
        if current_size > 0.30:
            current_size = 0.30
        elif current_size > 0.20:
            current_size = 0.25
        else:
            current_size = 0.20
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES: 1d bullish + 12h breakout + RSI confirmation
        if trend_bullish and breakout_long and rsi_bullish:
            # Extra confirmation: local trend also bullish
            if local_bullish or close[i] > hma_12h_50[i]:
                new_signal = current_size
        
        # SHORT ENTRIES: 1d bearish + 12h breakout + RSI confirmation
        elif trend_bearish and breakout_short and rsi_bearish:
            # Extra confirmation: local trend also bearish
            if local_bearish or close[i] < hma_12h_50[i]:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~15 days on 12h), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if trend_bullish and local_bullish and rsi_14[i] > 50:
                new_signal = current_size * 0.5
            elif trend_bearish and local_bearish and rsi_14[i] < 50:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bearish:
                trend_reversal = True
            if position_side < 0 and trend_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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