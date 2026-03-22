#!/usr/bin/env python3
"""
Experiment #089: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Complex regime-switching strategies failed because they had too many
conflicting conditions. This strategy uses a simpler, proven approach:
1. 1d HMA(21) slope for major trend bias (only trade WITH the daily trend)
2. 4h Donchian(20) breakout for entry timing (catch momentum moves)
3. 4h RSI(14) filter to avoid overextended entries (RSI 35-65 sweet spot)
4. ATR(14) trailing stop for risk management (2.5x ATR)

Why this should work:
- Donchian breakouts catch sustained moves (proven on SOL with Sharpe +0.782)
- 1d HTF filter prevents counter-trend trades (major issue in 2022 crash)
- RSI filter avoids chasing extended moves (reduces false breakouts)
- 4h timeframe naturally limits trades to 20-50/year
- Simpler logic = more trades = better statistics across all symbols

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (balanced risk/opportunity)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # 4h HMA for additional trend confirmation
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR FILTER) ===
        # Only trade in direction of daily trend
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === RSI FILTER (avoid overextended entries) ===
        # For longs: RSI should not be overbought (>70)
        # For shorts: RSI should not be oversold (<30)
        rsi_not_overbought = rsi_14[i] < 70
        rsi_not_oversold = rsi_14[i] > 30
        
        # RSI in sweet spot for entry (35-65)
        rsi_sweet_spot_long = 35 < rsi_14[i] < 65
        rsi_sweet_spot_short = 35 < rsi_14[i] < 65
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size if 1d trend is weak
        if not trend_1d_bullish and not trend_1d_bearish:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: 1d bullish + 4h bullish + Donchian breakout + RSI not overbought
        if trend_1d_bullish and hma_4h_bullish and breakout_long and rsi_not_overbought:
            new_signal = current_size
        # Secondary: 1d bullish + price above 1d HMA + Donchian breakout + RSI sweet spot
        elif trend_1d_bullish and price_above_1d_hma and breakout_long and rsi_sweet_spot_long:
            new_signal = current_size * 0.8
        # Tertiary (weaker): 1d bullish + 4h bullish + RSI pullback (no breakout needed)
        elif trend_1d_bullish and hma_4h_bullish and rsi_14[i] < 45:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        # Primary: 1d bearish + 4h bearish + Donchian breakout + RSI not oversold
        if trend_1d_bearish and hma_4h_bearish and breakout_short and rsi_not_oversold:
            new_signal = -current_size
        # Secondary: 1d bearish + price below 1d HMA + Donchian breakout + RSI sweet spot
        elif trend_1d_bearish and price_below_1d_hma and breakout_short and rsi_sweet_spot_short:
            new_signal = -current_size * 0.8
        # Tertiary (weaker): 1d bearish + 4h bearish + RSI rally (no breakout needed)
        elif trend_1d_bearish and hma_4h_bearish and rsi_14[i] > 55:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~25 days on 4h), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and price_above_1d_hma and rsi_14[i] < 50:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and price_below_1d_hma and rsi_14[i] > 50:
                new_signal = -current_size * 0.4
        
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
        # Exit if 1d trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and trend_1d_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and trend_1d_bullish:
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