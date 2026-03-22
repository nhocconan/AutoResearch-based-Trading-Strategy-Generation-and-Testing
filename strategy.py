#!/usr/bin/env python3
"""
Experiment #052: 12h Primary + 1d/1w HTF — Fisher Transform + Adaptive Trend

Hypothesis: Previous strategies failed due to over-filtering (too many confluence
conditions = 0 trades). This strategy SIMPLIFIES entry logic while keeping HTF bias:

1. 1w HMA(21) for MACRO trend bias (only trade WITH weekly trend)
2. 1d HMA(21) for INTERMEDIATE trend confirmation
3. Ehlers Fisher Transform(9) for entry timing (crosses -1.5/+1.5)
4. KAMA(14) for adaptive trend confirmation (ER-based smoothing)
5. ATR(14) for volatility-adjusted position sizing
6. Asymmetric entries: easier with trend, harder against

Why this should work:
- Fisher Transform catches reversals in bear/range markets (proven edge)
- 12h timeframe = 20-50 trades/year (optimal fee/trade balance)
- Weekly HMA prevents trading against macro trend
- KAMA adapts to volatility (smooths in trends, responsive in ranges)
- Looser Fisher thresholds (-1.5/+1.5 vs -1.0/+1.0) = more trades
- No session filter (12h bars span multiple sessions anyway)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete, vol-adjusted
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_kama_1d1w_hma_v1"
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

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    
    ER = |Change| / Sum(|Change|) over period
    High ER = trending (fast smoothing)
    Low ER = ranging (slow smoothing)
    """
    close_s = pd.Series(close)
    
    # Change over period
    change = close_s.diff(period).abs()
    
    # Sum of absolute changes over period
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for better signal detection
    
    Fisher = 0.5 * ln((1+X)/(1-X)) where X = 2*(price-LL)/(HH-LL) - 1
    
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Median price
    median = (high_s + low_s) / 2
    
    # Highest high and lowest low over period
    hh = median.rolling(window=period, min_periods=period).max()
    ll = median.rolling(window=period, min_periods=period).min()
    
    # Normalize to -1 to +1
    x = 2 * (median - ll) / (hh - ll).replace(0, np.nan) - 1
    x = x.clip(-0.999, 0.999)  # Prevent ln domain errors
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = fisher.fillna(0).values
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs moving average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std.replace(0, np.nan)
    return zscore.fillna(0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_14 = calculate_kama(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    zscore_20 = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15  # Against trend or high vol
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(kama_14[i]):
            continue
        
        # === WEEKLY TREND BIAS (MACRO) ===
        # Price above 1w HMA = bullish macro bias (prefer longs)
        # Price below 1w HMA = bearish macro bias (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY TREND CONFIRMATION (INTERMEDIATE) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === KAMA TREND CONFIRMATION ===
        # Price above KAMA = short-term bullish
        # Price below KAMA = short-term bearish
        kama_bullish = close[i] > kama_14[i]
        kama_bearish = close[i] < kama_14[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_short_signal = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Also allow continuation signals (Fisher already in extreme zone)
        fisher_long_zone = fisher[i] < -0.5
        fisher_short_zone = fisher[i] > 0.5
        
        # === VOLATILITY ADJUSTMENT ===
        # High vol = reduce position size
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[50:i+1]) if i > 50 else 1.0
        vol_adjustment = 1.0 if atr_ratio < 1.5 else 0.6
        
        # === POSITION SIZING ===
        # With trend = BASE_SIZE, Against trend = REDUCED_SIZE
        long_size = BASE_SIZE * vol_adjustment
        short_size = BASE_SIZE * vol_adjustment
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Weekly bullish + Daily bullish + Fisher long signal
        # Secondary: Weekly bullish + Fisher in long zone + KAMA bullish
        # Tertiary (loose): Fisher long signal + KAMA bullish (for trade frequency)
        
        if trend_1w_bullish and trend_1d_bullish and fisher_long_signal:
            new_signal = long_size
        elif trend_1w_bullish and fisher_long_zone and kama_bullish:
            new_signal = long_size
        elif fisher_long_signal and kama_bullish and bars_since_last_trade > 50:
            # Loose entry to ensure trade frequency
            new_signal = REDUCED_SIZE * vol_adjustment
        
        # SHORT ENTRIES
        # Primary: Weekly bearish + Daily bearish + Fisher short signal
        # Secondary: Weekly bearish + Fisher in short zone + KAMA bearish
        # Tertiary (loose): Fisher short signal + KAMA bearish (for trade frequency)
        
        if trend_1w_bearish and trend_1d_bearish and fisher_short_signal:
            new_signal = -short_size
        elif trend_1w_bearish and fisher_short_zone and kama_bearish:
            new_signal = -short_size
        elif fisher_short_signal and kama_bearish and bars_since_last_trade > 50:
            # Loose entry to ensure trade frequency
            new_signal = -REDUCED_SIZE * vol_adjustment
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~15 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher_long_zone:
                new_signal = REDUCED_SIZE * 0.5
            elif trend_1w_bearish and fisher_short_zone:
                new_signal = -REDUCED_SIZE * 0.5
        
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
            if position_side > 0 and trend_1w_bearish and fisher[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and fisher[i] < -1.0:
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