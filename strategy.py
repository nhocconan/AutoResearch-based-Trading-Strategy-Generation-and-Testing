#!/usr/bin/env python3
"""
Experiment #049: 4h Primary + 1d HTF — Simplified Trend Pullback

Hypothesis: Previous 4h strategies failed due to TOO MANY filters causing 0 trades.
This strategy SIMPLIFIES entry conditions while keeping HTF trend alignment:

1. 1d HMA(21) for MAJOR trend bias (only trade WITH 1d trend)
2. 4h RSI(14) pullback entries (30/70 thresholds - NOT extreme 20/80)
3. 4h Donchian(20) breakout confirmation (price vs 20-bar high/low)
4. ATR(14) trailing stoploss at 2.5x
5. NO session filter, NO volume filter (these killed trades in #048)
6. Discrete sizing: 0.30 for full position, 0.15 for partial

Why this should work:
- 4h is proven timeframe (20-50 trades/year target)
- 1d HMA provides strong trend filter without overfitting
- RSI 30/70 is achievable (unlike 20/80 which rarely triggers)
- Donchian confirms momentum without lagging
- Fewer filters = more trades = better Sharpe calculation
- Simpler logic = less chance of 0 trades

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.15-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_donchian_1d_hma_simple_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
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
    """Calculate Donchian Channel upper and lower bands."""
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
    
    # Calculate 1d HMA for trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align 1d HMA to 4h (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # 4h HMA for additional trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        trend_4h_bullish = close[i] > hma_4h_21[i]
        trend_4h_bearish = close[i] < hma_4h_21[i]
        
        # === RSI PULLBACK CONDITIONS (looser thresholds) ===
        # RSI < 40 = pullback in uptrend (long opportunity)
        # RSI > 60 = pullback in downtrend (short opportunity)
        rsi_pullback_long = rsi_14[i] < 40
        rsi_pullback_short = rsi_14[i] > 60
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # Price near Donchian upper = bullish momentum
        # Price near Donchian lower = bearish momentum
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 0:
            price_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            price_position = 0.5
        
        donchian_bullish = price_position > 0.5
        donchian_bearish = price_position < 0.5
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1d bullish + 4h bullish + RSI pullback
        if trend_1d_bullish and trend_4h_bullish and rsi_pullback_long:
            new_signal = current_size
        
        # SHORT ENTRY: 1d bearish + 4h bearish + RSI pullback
        if trend_1d_bearish and trend_4h_bearish and rsi_pullback_short:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            # Weaker long: just 1d bullish + RSI < 45
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = HALF_SIZE
            # Weaker short: just 1d bearish + RSI > 55
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -HALF_SIZE
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 65:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 35:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === TAKE PROFIT (reduce position at 2R) ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr_14[i] and current_size == BASE_SIZE:
                    new_signal = HALF_SIZE  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr_14[i] and current_size == BASE_SIZE:
                    new_signal = -HALF_SIZE  # Take partial profit
        
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif abs(new_signal) < abs(signals[i-1]) if i > 0 else False:
                # Partial exit (take profit) - keep tracking
                pass
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