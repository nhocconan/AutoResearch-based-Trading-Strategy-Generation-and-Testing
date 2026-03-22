#!/usr/bin/env python3
"""
Experiment #040: 1h Primary + 4h/12h HTF — Fisher Transform Mean Reversion

Hypothesis: Bear/range markets (2025 test period) require reversal strategies,
not trend following. Ehlers Fisher Transform excels at catching reversals in
bear market rallies. Combined with:

1. 12h HMA(21) for MAJOR trend bias (slow, fewer whipsaws)
2. 4h HMA(21) for INTERMEDIATE confirmation
3. Ehlers Fisher Transform(9) for entry timing (reversal signals)
4. Choppiness Index(14) regime filter (>55 = range, <45 = trend)
5. Volume filter (>0.8x 20-bar avg)
6. Session filter (8-20 UTC only - high liquidity)
7. ATR(14) trailing stoploss at 2.5x

Why this should work:
- Fisher Transform has superior reversal detection vs RSI in bear markets
- 12h HMA slower than 4h = fewer trend whipsaws
- 1h entries within 12h/4h trend = HTF frequency with LTF precision
- Choppiness prevents trend-following in ranges (2025 is range/bear)
- Session filter avoids low-liquidity whipsaws
- Discrete sizing (0.20/0.30) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_4h12h_hma_v1"
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
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - superior reversal detection for bear markets.
    
    Steps:
    1. Calculate typical price: (2*close + high + low) / 4
    2. Normalize to -1 to +1 range
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    
    Entry signals:
    - Fisher crosses above -1.5 from below = long reversal
    - Fisher crosses below +1.5 from above = short reversal
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (2 * close_s + high_s + low_s) / 4
    
    # Normalize to -1 to +1 (using highest high - lowest low over period)
    hh = typical.rolling(window=period, min_periods=period).max()
    ll = typical.rolling(window=period, min_periods=period).min()
    range_hl = hh - ll
    range_hl = range_hl.replace(0, np.nan)
    
    normalized = (2 * (typical - ll) / range_hl) - 1
    normalized = normalized.clip(-0.999, 0.999)  # Avoid ln(0) or ln(inf)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Smooth with EMA
    fisher_smooth = fisher.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Fisher trigger (1-period EMA of fisher)
    fisher_trigger = fisher_smooth.ewm(span=1, min_periods=1, adjust=False).mean()
    
    return fisher_smooth.values, fisher_trigger.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
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
    hh_ll = np.where(hh_ll == 0, np.nan, hh_ll)
    
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            if not in_position:
                signals[i] = 0.0
                prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
                continue
        
        # === 12H TREND BIAS (MAJOR - SLOW) ===
        # Price above 12h HMA = bullish bias (prefer longs)
        # Price below 12h HMA = bearish bias (prefer shorts)
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion preferred)
        # CHOP < 45 = trend (trend following preferred)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long_signal = (prev_fisher < -1.5) and (fisher[i] >= -1.5)
        fisher_short_signal = (prev_fisher > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow extreme Fisher values without cross (stronger signal)
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === RSI CONFIRMATION ===
        # RSI < 35 confirms oversold for longs
        # RSI > 65 confirms overbought for shorts
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 12h bullish OR (range regime + Fisher extreme)
        # Plus: 4h confirmation + volume + Fisher reversal + RSI confirmation
        if trend_12h_bullish or (is_range and fisher_extreme_long):
            if trend_4h_bullish and volume_ok:
                if fisher_long_signal or fisher_extreme_long:
                    if rsi_oversold or is_range:
                        new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 12h bearish OR (range regime + Fisher extreme)
        # Plus: 4h confirmation + volume + Fisher reversal + RSI confirmation
        if trend_12h_bearish or (is_range and fisher_extreme_short):
            if trend_4h_bearish and volume_ok:
                if fisher_short_signal or fisher_extreme_short:
                    if rsi_overbought or is_range:
                        new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~12 days on 1h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and trend_4h_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.6
            elif trend_12h_bearish and trend_4h_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.6
        
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
            if position_side > 0 and trend_12h_bearish and fisher[i] > 1.5:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and fisher[i] < -1.5:
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
        
        # Update prev_fisher for next iteration
        prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
    
    return signals