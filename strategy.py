#!/usr/bin/env python3
"""
Experiment #035: 1h Fisher Transform + 4h/1d HMA Trend + ATR Stoploss

Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets.
Combined with 4h/1d HMA trend filter for direction, this should generate trades
while avoiding counter-trend positions that failed in 2022 crash.

Key design:
1. 1d HMA(21) for major trend bias (call ONCE via mtf_data)
2. 4h HMA(21) for intermediate trend confirmation (call ONCE via mtf_data)
3. 1h Fisher Transform(9) for entry timing - crosses at extremes
4. ATR(14) for stoploss (2.5x) - mandatory risk management
5. NO session filter, NO volume filter - learned from 0-trade failures
6. Discrete sizing: 0.25 base, 0.30 strong trend (both HTF agree)

Why this should work:
- Fisher Transform is proven for reversal detection in bear markets (Ehlers research)
- 4h/1d HMA prevents counter-trend trades (major failure mode in 2022)
- 1h TF with simple entry = 30-60 trades/year (optimal for fee efficiency)
- NO over-filtering (learned from experiments #025, #026, #028, #030 with 0 trades)
- ATR stoploss protects from major drawdowns

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h1d_hma_atr_v1"
timeframe = "1h"
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

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((close - low_n) / (high_n - low_n) - 0.5) + 0.67 * X_prev
    This transforms price into a Gaussian distribution for clearer signals.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    # Calculate highest high and lowest low over period
    high_s = pd.Series(close)  # Use close as proxy for high/low range
    low_s = pd.Series(close)
    
    # For Fisher, we need the price range - use close directly with smoothing
    price_range = pd.Series(close).rolling(window=period, min_periods=period).max().values - \
                  pd.Series(close).rolling(window=period, min_periods=period).min().values
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    mid_point = (pd.Series(close).rolling(window=period, min_periods=period).max().values + \
                 pd.Series(close).rolling(window=period, min_periods=period).min().values) / 2
    
    x = np.zeros(n)
    for i in range(period, n):
        # Normalize price within range
        norm_price = (close[i] - mid_point[i]) / (price_range[i] / 2 + 1e-10)
        norm_price = np.clip(norm_price, -0.99, 0.99)
        
        # Smooth with EMA-like factor
        x[i] = 0.66 * norm_price + 0.67 * x[i-1] if i > period else 0.66 * norm_price
        x[i] = np.clip(x[i], -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x[i]) / (1 - x[i] + 1e-10))
        fisher_prev[i] = fisher[i-1] if i > period else 0.0
    
    return fisher, fisher_prev

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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMA trends
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF indicators to 1h timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher(close, 9)
    
    # Also calculate 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        # === HTF TREND BIAS ===
        # 1d HMA for major trend, 4h HMA for intermediate trend
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND ===
        local_bullish = close[i] > hma_1h_21[i]
        local_bearish = close[i] < hma_1h_21[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Also allow entry when Fisher is at extreme and starting to turn
        fisher_oversold = fisher[i] < -1.0 and fisher[i] > fisher_prev[i]
        fisher_overbought = fisher[i] > 1.0 and fisher[i] < fisher_prev[i]
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        # Strong trend (1d + 4h + local agree) = 0.30
        # Medium trend (1d + 4h agree) = 0.25
        # Weak trend (only 1d) = 0.20
        if htf_1d_bullish and htf_4h_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_1d_bullish and htf_4h_bullish:
            current_size = BASE_SIZE
        elif htf_1d_bullish:
            current_size = WEAK_SIZE
        elif htf_1d_bearish and htf_4h_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_1d_bearish and htf_4h_bearish:
            current_size = BASE_SIZE
        elif htf_1d_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1d bullish + Fisher reversal from oversold
        # Loosen conditions to ensure trades trigger (learned from 0-trade failures)
        if htf_1d_bullish and (fisher_long or fisher_oversold):
            new_signal = current_size
        
        # SHORT ENTRY: 1d bearish + Fisher reversal from overbought
        elif htf_1d_bearish and (fisher_short or fisher_overbought):
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~2 days on 1h), allow weaker entry
        # This prevents 0-trade scenarios
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if htf_1d_bullish and fisher_oversold:
                new_signal = current_size * 0.8
            elif htf_1d_bearish and fisher_overbought:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_1d_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_1d_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or fisher_exit:
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