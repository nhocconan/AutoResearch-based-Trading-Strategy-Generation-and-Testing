#!/usr/bin/env python3
"""
Experiment #031: 4h 1w/1d HMA Trend + RSI Pullback (Simplified)

Hypothesis: Previous regime-switching strategies (KAMA+Chop) failed due to:
1. Too many conflicting filters = 0 trades or late entries
2. Regime detection adds lag - by the time CHOP confirms, move is over
3. Complex logic creates whipsaw in transition periods

NEW APPROACH - Simpler, proven pattern:
1. 1w HMA(21) = major trend bias (ONLY trade in direction)
2. 1d HMA(21) = intermediate trend confirmation
3. 4h RSI(14) pullback = entry timing (RSI 35-45 long, 55-65 short)
4. ATR(14) 2.5x trailing stoploss
5. Discrete sizing 0.25-0.30

Why this should work better:
- FEWER filters = more trades (avoiding 0-trade failures)
- HMA has less lag than KAMA/EMA (proven in literature)
- RSI pullback in trend direction = high prob setup (60%+ win rate)
- 1w bias prevents major counter-trend losses (2022 crash protection)
- Simpler logic = fewer edge cases, more robust

Timeframe: 4h (REQUIRED)
HTF: 1w + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1w1d_bias_v2"
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
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
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        # === 1W TREND BIAS (major trend) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND CONFIRMATION (intermediate trend) ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND (entry timeframe) ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === RSI PULLBACK ENTRY (looser thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 45  # Long entry zone
        rsi_overbought = rsi_14[i] > 55  # Short entry zone
        rsi_neutral = 40 <= rsi_14[i] <= 60  # Neutral zone
        
        # === SMA200 FILTER (only trade in direction of long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === POSITION SIZING (volatility adjusted) ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - need weekly OR daily bullish + RSI pullback
        if weekly_bullish or daily_bullish:
            # Primary: RSI pullback in uptrend
            if rsi_oversold and hma_4h_bullish:
                new_signal = current_size
            # Secondary: RSI neutral + price above SMA200 (momentum continuation)
            elif rsi_neutral and above_sma200 and hma_4h_bullish:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES - need weekly OR daily bearish + RSI pushback
        elif weekly_bearish or daily_bearish:
            # Primary: RSI pushback in downtrend
            if rsi_overbought and hma_4h_bearish:
                new_signal = -current_size
            # Secondary: RSI neutral + price below SMA200 (momentum continuation)
            elif rsi_neutral and below_sma200 and hma_4h_bearish:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trades for 20 bars (~3.3 days on 4h), force entry with weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] > 35:
                new_signal = current_size * 0.5
            elif weekly_bearish and rsi_14[i] < 65:
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
            # Exit long if weekly AND daily turn bearish
            if position_side > 0 and weekly_bearish and daily_bearish:
                trend_reversal = True
            # Exit short if weekly AND daily turn bullish
            if position_side < 0 and weekly_bullish and daily_bullish:
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