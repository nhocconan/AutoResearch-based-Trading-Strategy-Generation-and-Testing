#!/usr/bin/env python3
"""
Experiment #046: 12h Primary + 1d HTF — Simplified Dual Regime

Hypothesis: Previous strategies failed due to too many conflicting filters
(Choppiness + Connors RSI + multiple HTF = 0 trades). This strategy SIMPLIFIES:

1. 1d HMA(21) for trend bias (loaded ONCE via mtf_data helper)
2. 12h RSI(7) for faster entry signals (not extreme 15/85, use 30/70)
3. Bollinger Band position for mean-reversion context
4. ATR(14) trailing stoploss at 2.5x
5. Looser thresholds to ensure 30+ trades per symbol

Why this should work:
- 12h timeframe = ~30-50 trades/year target (proven range)
- 1d HMA trend filter prevents counter-trend disasters in 2022 crash
- RSI(7) responds faster than RSI(14) for entry timing
- BB position adds confluence without blocking all trades
- Simple logic = fewer conditions that can all fail simultaneously

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_bb_1d_hma_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # BB position: 0=lower, 0.5=middle, 1=upper
    bb_width = upper - lower
    bb_position = (close - lower) / bb_width.replace(0, np.nan)
    
    return sma.values, upper.values, lower.values, bb_position.fillna(0.5).values

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
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entries
    rsi_14 = calculate_rsi(close, 14)  # Standard RSI for confirmation
    bb_sma, bb_upper, bb_lower, bb_position = calculate_bollinger_bands(close, 20, 2.0)
    hma_12h_21 = calculate_hma(close, 21)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_position[i]):
            continue
        
        # === 1D TREND BIAS ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_bullish = close[i] > hma_1d_21_aligned[i]
        trend_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA CONFIRMATION ===
        hma_bullish = close[i] > hma_12h_21[i]
        hma_bearish = close[i] < hma_12h_21[i]
        
        # === RSI SIGNALS (looser thresholds for more trades) ===
        rsi_oversold = rsi_7[i] < 35  # Was 30, loosened for more trades
        rsi_overbought = rsi_7[i] > 65  # Was 70, loosened for more trades
        
        # === BOLLINGER BAND POSITION ===
        bb_low = bb_position[i] < 0.2  # Near lower band
        bb_high = bb_position[i] > 0.8  # Near upper band
        bb_mid = bb_position[i] > 0.4 and bb_position[i] < 0.6  # Near middle
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require bullish 1d trend OR strong mean-reversion)
        if trend_bullish:
            # Trend-following long: RSI pullback + BB support
            if rsi_oversold and (bb_low or bb_mid):
                new_signal = current_size
            # Strong trend: RSI not extreme but HMA support
            elif rsi_7[i] < 45 and hma_bullish and bb_position[i] < 0.5:
                new_signal = current_size * 0.7
        elif trend_bearish:
            # Counter-trend long only on extreme oversold (mean reversion)
            if rsi_7[i] < 25 and bb_low:
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES (require bearish 1d trend OR strong mean-reversion)
        if trend_bearish:
            # Trend-following short: RSI pullback + BB resistance
            if rsi_overbought and (bb_high or bb_mid):
                new_signal = -current_size
            # Strong trend: RSI not extreme but HMA resistance
            elif rsi_7[i] > 55 and hma_bearish and bb_position[i] > 0.5:
                new_signal = -current_size * 0.7
        elif trend_bullish:
            # Counter-trend short only on extreme overbought (mean reversion)
            if rsi_7[i] > 75 and bb_high:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~15 days on 12h), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if trend_bullish and rsi_7[i] < 40:
                new_signal = current_size * 0.5
            elif trend_bearish and rsi_7[i] > 60:
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
            if position_side > 0 and trend_bearish and rsi_14[i] > 60:
                trend_reversal = True
            if position_side < 0 and trend_bullish and rsi_14[i] < 40:
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