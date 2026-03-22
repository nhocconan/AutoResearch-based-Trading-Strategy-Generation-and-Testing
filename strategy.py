#!/usr/bin/env python3
"""
Experiment #069: 4h Primary + 1d HTF — Donchian Breakout Trend Following

Hypothesis: Previous strategies failed due to overly complex regime filters (Connors+Chop).
This strategy uses PROVEN patterns from research that worked on SOL/ETH:

1. 1d HMA(21) SLOPE for major trend direction (simple, effective)
2. 4h Donchian(20) breakout for entries (proven Sharpe +0.782 on SOL)
3. RSI(14) filter with 45/55 thresholds (achievable, not extreme)
4. ATR(14) trailing stop at 2.5x (standard risk management)
5. Position size: 0.30 discrete (balanced risk/opportunity)

Why this should work:
- Donchian breakouts naturally generate 20-50 trades/year on 4h
- 1d HMA slope prevents counter-trend trades in strong trends
- RSI 45/55 thresholds are achievable (not 20/80 extremes)
- Simpler logic = more trades = better statistics
- ATR trailing stop captures trends while protecting capital

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_v1"
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Donchian Channel (20 period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # HMA for additional trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_4h_21[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H PRICE POSITION ===
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper channel = bullish signal
        # Breakout below lower channel = bearish signal
        # Check if price JUST broke out (was inside previous bar)
        donchian_breakout_long = (close[i] > donchian_upper[i] and 
                                   close[i-1] <= donchian_upper[i-1])
        donchian_breakout_short = (close[i] < donchian_lower[i] and 
                                    close[i-1] >= donchian_lower[i-1])
        
        # Also allow entries when price is near breakout level with momentum
        near_upper = close[i] > donchian_upper[i] * 0.98
        near_lower = close[i] < donchian_lower[i] * 1.02
        
        # === RSI MOMENTUM FILTER ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # Entry thresholds (achievable, not extreme)
        rsi_entry_long = rsi_14[i] > 45
        rsi_entry_short = rsi_14[i] < 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size when trend is weak (slope near zero)
        if abs(hma_1d_slope_aligned[i]) < 1.0:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Donchian breakout + 1d bullish + RSI confirmation
        if trend_1d_bullish or price_above_1d_hma:
            if donchian_breakout_long and rsi_entry_long:
                new_signal = current_size
            # Secondary: Near breakout with strong momentum
            elif near_upper and rsi_14[i] > 55 and price_above_4h_hma:
                new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        # Primary: Donchian breakout + 1d bearish + RSI confirmation
        if trend_1d_bearish or price_below_1d_hma:
            if donchian_breakout_short and rsi_entry_short:
                new_signal = -current_size
            # Secondary: Near breakout with strong momentum
            elif near_lower and rsi_14[i] < 45 and price_below_4h_hma:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~33 days on 4h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and price_above_4h_hma and rsi_14[i] > 50:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and price_below_4h_hma and rsi_14[i] < 50:
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
            # Exit long if 1d trend reverses bearish
            if position_side > 0 and trend_1d_bearish and price_below_4h_hma:
                trend_reversal = True
            # Exit short if 1d trend reverses bullish
            if position_side < 0 and trend_1d_bullish and price_above_4h_hma:
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