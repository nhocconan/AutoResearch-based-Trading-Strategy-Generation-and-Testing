#!/usr/bin/env python3
"""
Experiment #033: 1d Donchian Breakout + Weekly HMA Trend + RSI Filter

Hypothesis: Daily timeframe with weekly bias provides stable trend direction while
Donchian breakouts capture momentum moves. Key lessons from 29 failed experiments:
1. LOOSE entry conditions are critical (many strategies got 0 trades)
2. Weekly HMA(21) provides stable major trend filter without whipsaw
3. Donchian(20) breakout captures sustained moves on daily
4. RSI thresholds must be wide (40-60 range, not narrow bands)
5. ATR trailing stop at 2.5x protects capital in 2022-style crashes
6. Position size 0.30 discrete minimizes fee churn while capturing moves

Why this should work on 1d:
- 1d naturally produces 20-50 trades/year (perfect for fee management)
- Weekly bias prevents counter-trend trades in major crashes
- Donchian breakout is proven momentum indicator (Turtle Trading)
- Loose RSI filter ensures trades actually happen (avoiding 0-trade failure)
- Simple logic = fewer conflicting conditions = more consistent signals

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-40 trades per symbol over train period
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_hma_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === WEEKLY TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_50[i]
        daily_bearish = close[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper channel
        breakout_long = close[i] > donchian_upper[i]
        # Breakout below lower channel
        breakout_short = close[i] < donchian_lower[i]
        
        # === RSI CONFIRMATION (LOOSE thresholds to ensure trades) ===
        rsi_ok_long = rsi_14[i] > 40  # Very loose - just not oversold
        rsi_ok_short = rsi_14[i] < 60  # Very loose - just not overbought
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (LOOSE conditions to ensure trades happen) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Weekly bullish + Donchian breakout + RSI confirmation
        if weekly_bullish and breakout_long and rsi_ok_long:
            new_signal = current_size
        # Alternative: Daily bullish + Donchian breakout (when weekly neutral)
        elif daily_bullish and breakout_long and rsi_ok_long and not weekly_bearish:
            new_signal = current_size * 0.8
        
        # SHORT ENTRIES - Weekly bearish + Donchian breakout + RSI confirmation
        elif weekly_bearish and breakout_short and rsi_ok_short:
            new_signal = -current_size
        # Alternative: Daily bearish + Donchian breakout (when weekly neutral)
        elif daily_bearish and breakout_short and rsi_ok_short and not weekly_bullish:
            new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), force entry with weaker conditions
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.5
            elif weekly_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.5
        
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
            if position_side > 0 and weekly_bearish:
                # Weekly trend turned bearish - exit long
                trend_reversal = True
            if position_side < 0 and weekly_bullish:
                # Weekly trend turned bullish - exit short
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === DONCHIAN EXIT (opposite breakout) ===
        if in_position and position_side > 0 and breakout_short:
            # Long position + short breakout = exit
            new_signal = 0.0
        
        if in_position and position_side < 0 and breakout_long:
            # Short position + long breakout = exit
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
            # If same direction, keep position (no update needed)
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