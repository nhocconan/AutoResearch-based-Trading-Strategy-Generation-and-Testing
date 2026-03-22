#!/usr/bin/env python3
"""
Experiment #033: 1d HMA-Donchian with 1w Trend Bias

Hypothesis: Higher timeframe (1d) with weekly trend filter reduces whipsaw while
maintaining sufficient trade frequency. Previous 1d strategies failed due to:
- Too many confluence requirements (0 trades)
- Overly complex regime detection
- Wrong position sizing (too aggressive)

This strategy simplifies to core signals:
1. 1w HMA(21) = major trend bias (long above, short below)
2. 1d HMA(16/48) crossover = trend momentum
3. Donchian(20) breakout = entry trigger
4. RSI(14) filter = avoid extreme entries
5. 2.5 ATR trailing stop = risk management

Key improvements from failed experiments:
- Fewer entry conditions (2-3 required, not 4+)
- Discrete position sizing (0.25, 0.30) to reduce fee churn
- Proper MTF alignment using mtf_data helper (call ONCE before loop)
- Looser entry when no trades for 45+ bars (frequency safeguard)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() - called ONCE before loop
Position sizing: 0.25-0.30 (discrete levels)
Stoploss: 2.5 * ATR(14) trailing
Target: 20-50 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_donchian_1w_bias_rsi_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

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
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1D indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_16 = calculate_hma(close, 16)  # Faster HMA for entry signal
    hma_1d_48 = calculate_hma(close, 48)  # Slower HMA for trend
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50  # Track last trade for frequency control
    
    for i in range(200, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W TREND BIAS (major trend direction) ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HMA TREND ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === HMA TREND STRENGTH (slope) ===
        hma_slope_long = hma_1d_16[i] > hma_1d_16[i-3] if i > 3 else False
        hma_slope_short = hma_1d_16[i] < hma_1d_16[i-3] if i > 3 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi_14[i] < 70  # Don't long at extreme overbought
        rsi_ok_short = rsi_14[i] > 30  # Don't short at extreme oversold
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === POSITION SIZING (discrete levels) ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED - 2-3 conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need weekly bias + daily trend + breakout
        # Core: weekly_bullish + hma_bullish (2 required)
        # Optional: breakout_long OR above_sma200 (1 of 2)
        long_core = weekly_bullish and hma_bullish
        long_optional = breakout_long or above_sma200
        
        if long_core and long_optional and rsi_ok_long:
            new_signal = current_size
        
        # SHORT ENTRY: Need weekly bias + daily trend + breakout
        short_core = weekly_bearish and hma_bearish
        short_optional = breakout_short or below_sma200
        
        if short_core and short_optional and rsi_ok_short:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), loosen entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            # Allow entry with just weekly + daily trend (no breakout required)
            if weekly_bullish and hma_bullish and rsi_14[i] < 65:
                new_signal = current_size * 0.8  # Slightly smaller size
            elif weekly_bearish and hma_bearish and rsi_14[i] > 35:
                new_signal = -current_size * 0.8
        
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
            # Exit long if daily HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if daily HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === WEEKLY BIAS REVERSAL EXIT ===
        bias_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly turns bearish
            if position_side > 0 and weekly_bearish:
                bias_reversal = True
            # Exit short if weekly turns bullish
            if position_side < 0 and weekly_bullish:
                bias_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or bias_reversal:
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals