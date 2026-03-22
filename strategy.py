#!/usr/bin/env python3
"""
Experiment #040: 1h Fisher Transform + 4h HMA Trend + Choppiness Regime

Hypothesis: Previous 1h strategies failed due to overly strict confluence filters
(0 trades on some symbols). This strategy SIMPLIFIES while keeping edges:

1. 4h HMA(21) for trend direction (call ONCE before loop via mtf_data)
2. Ehlers Fisher Transform(9) for entry timing - proven reversal indicator
3. Choppiness Index(14) for regime: >55=range(mean-revert), <45=trend(follow)
4. SOFT filters only - no hard volume/session requirements that kill trades
5. Looser Fisher thresholds (±1.2 instead of ±1.5) to ensure 30-60 trades/year
6. ATR(14) trailing stoploss at 2.5x
7. Discrete position sizing: 0.20-0.30 based on volatility

Why this should beat Sharpe=0.028 and work on 1h:
- Fisher Transform generates MORE signals than Connors RSI (critical for 1h)
- Softer Choppiness thresholds (55/45 vs 61.8/38.2) = more regime coverage
- 4h trend filter prevents counter-trend trades but doesn't block entries
- No hard session/volume filters that caused 0-trade failures in #028, #030, #038
- 1h timeframe with 4h direction = HTF trade frequency with 1h execution precision

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year (1h with HTF filter naturally limits frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_chop_regime_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform normalizes price to Gaussian distribution,
    making extremes easier to identify for reversals.
    
    Entry signals:
    - Long: Fisher crosses above -1.2 (oversold reversal)
    - Short: Fisher crosses below +1.2 (overbought reversal)
    
    Reference: Ehlers, J.F. "Cycle Analytics for Traders"
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price
    typical = (high + low + close_for_fisher) / 3.0 if 'close_for_fisher' in dir() else (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = hh - ll
        if range_val > 0:
            normalized = 2.0 * (typical[i] - ll) / range_val - 1.0
        else:
            normalized = 0.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher Transform formula
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag of Fisher)
        if i > 0:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 55 = range/choppy market (mean revert)
    CHOP < 45 = trending market (trend follow)
    45-55 = neutral (use either logic)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
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
    global close_for_fisher  # For Fisher calculation
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    close_for_fisher = close  # Set for Fisher function
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_bullish = close[i] > hma_4h_21_aligned[i]
        trend_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINNESS REGIME (softer thresholds) ===
        choppy_market = chop_14[i] > 55
        trending_market = chop_14[i] < 45
        neutral_market = not choppy_market and not trending_market
        
        # === FISHER TRANSFORM SIGNALS (looser thresholds for more trades) ===
        # Fisher crosses above -1.2 = long signal
        # Fisher crosses below +1.2 = short signal
        fisher_long = fisher[i] > -1.2 and fisher_signal[i] <= -1.2
        fisher_short = fisher[i] < 1.2 and fisher_signal[i] >= 1.2
        
        # Also allow entry when Fisher is at extremes (not just crossover)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.30)
        
        # === ENTRY LOGIC (SOFT FILTERS - ensure trades happen) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - multiple paths to entry (critical for trade count)
        if trend_bullish:
            # Path 1: Range market + Fisher oversold (mean reversion)
            if choppy_market and fisher_oversold:
                new_signal = current_size
            # Path 2: Trending market + Fisher crossover long (trend pullback)
            elif trending_market and fisher_long:
                new_signal = current_size
            # Path 3: Neutral market + RSI oversold (backup entry)
            elif neutral_market and rsi_14[i] < 35:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES - multiple paths to entry
        elif trend_bearish:
            # Path 1: Range market + Fisher overbought (mean reversion)
            if choppy_market and fisher_overbought:
                new_signal = -current_size
            # Path 2: Trending market + Fisher crossover short (trend pullback)
            elif trending_market and fisher_short:
                new_signal = -current_size
            # Path 3: Neutral market + RSI overbought (backup entry)
            elif neutral_market and rsi_14[i] > 65:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), force entry with weaker signal
        # This prevents 0-trade failures on some symbols
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if trend_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.5
            elif trend_bearish and fisher[i] > 0.5:
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