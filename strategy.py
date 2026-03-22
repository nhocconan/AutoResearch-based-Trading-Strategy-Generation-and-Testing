#!/usr/bin/env python3
"""
Experiment #029: 4h Primary + 1d HTF — Simplified Trend Following with Regime Filter

Hypothesis: Previous failures due to overly complex filters preventing trades.
This strategy uses PROVEN patterns from research with SIMPLIFIED logic:

1. 1d HMA(21) for MAJOR trend bias (only trade WITH daily trend)
2. 4h HMA(21/48) crossover for entry timing
3. RSI(14) filter to avoid extreme overbought/oversold entries
4. Choppiness Index(14) to avoid range-bound whipsaws (>55 = skip)
5. ATR(14) trailing stoploss at 2.5x for risk management
6. Discrete sizing: 0.30 for entry, 0.15 for partial take-profit

Why this should work:
- 4h timeframe naturally limits trades to 20-50/year (proven in research)
- HMA crossover is smoother than EMA, less whipsaw
- 1d trend filter prevents counter-trend trades in strong moves
- Choppiness filter avoids 2022-style range destruction
- Simpler logic = more trades meeting minimum threshold
- ATR stoploss protects from 2022-style crashes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 entry, 0.15 partial exit
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_1d_regime_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy (avoid trend trades)
    CHOP < 38.2 = trending (prefer trend trades)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR(1) = True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over period
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # ATR(period)
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)  # avoid division by zero
    
    # Choppiness Index
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # HMA crossover signals (21 vs 48)
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    ENTRY_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position state for stoploss and take-profit
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    take_profit_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 55 = range/choppy (reduce position or skip new entries)
        # CHOP < 45 = trending (full position allowed)
        is_choppy = chop[i] > 55
        is_trending = chop[i] < 45
        
        # === HMA CROSSOVER SIGNAL ===
        # Bullish: HMA21 crosses above HMA48
        # Bearish: HMA21 crosses below HMA48
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        
        # Check for crossover (previous bar was opposite)
        hma_cross_long = hma_bullish and (i > 0 and hma_21[i-1] <= hma_48[i-1])
        hma_cross_short = hma_bearish and (i > 0 and hma_21[i-1] >= hma_48[i-1])
        
        # === RSI FILTER ===
        # Avoid entering when RSI is extreme (overbought for long, oversold for short)
        rsi_not_overbought = rsi_14[i] < 70
        rsi_not_oversold = rsi_14[i] > 30
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        # Require: 1d bullish trend + HMA cross long + RSI not overbought
        # In choppy market: only enter on strong crossover
        if trend_1d_bullish and hma_cross_long and rsi_not_overbought:
            if is_trending or (is_choppy and bars_since_last_trade > 50):
                new_signal = ENTRY_SIZE
        
        # SHORT ENTRY
        # Require: 1d bearish trend + HMA cross short + RSI not oversold
        # In choppy market: only enter on strong crossover
        if trend_1d_bearish and hma_cross_short and rsi_not_oversold:
            if is_trending or (is_choppy and bars_since_last_trade > 50):
                new_signal = -ENTRY_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and hma_bullish and rsi_14[i] < 60:
                new_signal = ENTRY_SIZE * 0.5
            elif trend_1d_bearish and hma_bearish and rsi_14[i] > 40:
                new_signal = -ENTRY_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        take_profit_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for trailing stop
                if close[i] > highest_price:
                    highest_price = close[i]
                
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
                
                # Take profit at 2R (reduce to half size)
                if not take_profit_hit:
                    profit_target = entry_price + 2 * 2.5 * atr_14[i]
                    if close[i] > profit_target:
                        take_profit_triggered = True
                        take_profit_hit = True
            
            if position_side < 0:
                # Update lowest price for trailing stop
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
                
                # Take profit at 2R (reduce to half size)
                if not take_profit_hit:
                    profit_target = entry_price - 2 * 2.5 * atr_14[i]
                    if close[i] < profit_target:
                        take_profit_triggered = True
                        take_profit_hit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and hma_bearish:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and hma_bullish:
                trend_reversal = True
        
        # === CHOPPINESS EXIT ===
        # If market becomes very choppy while in position, reduce exposure
        choppy_exit = False
        if in_position and is_choppy and chop[i] > 65:
            choppy_exit = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered:
            new_signal = 0.0
            take_profit_hit = False
        elif take_profit_triggered:
            new_signal = HALF_SIZE * np.sign(position_side)
        elif trend_reversal or choppy_exit:
            new_signal = 0.0
            take_profit_hit = False
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                take_profit_hit = False
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                take_profit_hit = False
            elif abs(new_signal) < abs(signals[i-1]) if i > 0 else True:
                # Partial exit (take profit)
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
                take_profit_hit = False
        
        signals[i] = new_signal
    
    return signals