#!/usr/bin/env python3
"""
Experiment #048: 30m Primary + 4h/1d HTF — Fisher Transform + Regime Confluence

Hypothesis: Previous 30m strategies failed due to (1) too many trades, (2) wrong 
indicator for bear/range markets. This strategy uses:

1. Ehlers Fisher Transform (period=9) - catches reversals in bear rallies (proven edge)
2. 1d HMA(21) - MAJOR trend bias (only trade WITH 1d trend direction)
3. 4h Choppiness Index(14) - regime filter (>55 = range mean-revert, <45 = trend follow)
4. 30m Bollinger Band Width percentile - volatility squeeze breakout confirmation
5. ADX(14) > 20 - confirms genuine momentum (filters choppy noise)
6. Session filter (10-18 UTC only - peak liquidity, avoids Asia/US overnight whipsaws)
7. Volume > 1.0x 20-bar avg - confirms genuine moves
8. Minimum 150 bars (~3 days) between trades - forces selectivity

Why Fisher Transform: Unlike RSI which lags, Fisher normalizes price to Gaussian 
distribution, creating sharper reversal signals. Works exceptionally well in 
bear/range markets (2022 crash, 2025 test period). Entry when Fisher crosses 
above -1.5 (long) or below +1.5 (short).

Why this should beat Connors RSI approaches:
- Fisher has faster reaction to reversals (critical for 30m TF)
- BB Width squeeze + ADX confirms breakout validity (reduces false signals)
- Tighter session filter (10-18 vs 8-20) avoids low-liquidity periods
- Minimum trade spacing prevents overtrading

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol (strict confluence = fewer but higher quality)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_chop_bb_4h1d_hma_v1"
timeframe = "30m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - lowest) / (highest - lowest) - 0.5) + 0.67 * prev_normalized
    3. Fisher = 0.5 * ln((1 + normalized) / (1 - normalized))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 from below
    - Short: Fisher crosses below +1.5 from above
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Typical price
    typical_price = (high_s + low_s) / 2
    
    # Normalize price within lookback window
    normalized = np.zeros(len(close))
    
    for i in range(period, len(close)):
        highest = high_s.iloc[i-period+1:i+1].max()
        lowest = low_s.iloc[i-period+1:i+1].min()
        
        if highest > lowest:
            raw_norm = 0.66 * ((typical_price.iloc[i] - lowest) / (highest - lowest) - 0.5)
            if i > period:
                raw_norm += 0.67 * normalized[i-1]
            # Clamp to prevent division issues
            normalized[i] = np.clip(raw_norm, -0.99, 0.99)
    
    # Fisher Transform
    fisher = np.zeros(len(close))
    for i in range(period, len(close)):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i] + 1e-10))
    
    return fisher

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
    
    # Choppiness Index
    chop = 100 * (atr1_sum / atr_period) / (hh_ll + 1e-10) * np.log10(period)
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

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width percentile for squeeze detection."""
    close_s = pd.Series(close)
    
    # Bollinger Bands
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Band Width
    bb_width = (upper - lower) / sma
    
    # Percentile rank over 60 bars (2 days on 30m)
    bb_width_pct = bb_width.rolling(window=60, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 60 else 0.5
    )
    
    return bb_width_pct.values

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    chop_4h = calculate_choppiness_index(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_width_pct = calculate_bb_width(close, 20, 2.0)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40, lower for 30m)
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -150
    
    # Track Fisher crossings
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(chop_4h_aligned[i]) or np.isnan(adx_14[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        prev_fisher_val = prev_fisher
        prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
        
        if np.isnan(fisher[i]):
            continue
        
        # === SESSION FILTER (10-18 UTC only - peak liquidity) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 10 <= utc_hour <= 18
        
        if not in_session:
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === 1D TREND BIAS (MAJOR) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion preferred)
        # CHOP < 45 = trend (trend following preferred)
        is_range = chop_4h_aligned[i] > 55
        is_trend = chop_4h_aligned[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 1.0 * volume_sma[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20
        
        # === BB WIDTH SQUEEZE ===
        # Low percentile = squeeze (potential breakout)
        bb_squeeze = bb_width_pct[i] < 0.30
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        # Short: Fisher crosses below +1.5 from above
        fisher_long_cross = (fisher[i] > -1.5) and (prev_fisher_val <= -1.5)
        fisher_short_cross = (fisher[i] < 1.5) and (prev_fisher_val >= 1.5)
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in range regime (more cautious)
        if is_range:
            current_size = BASE_SIZE * 0.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Minimum bars between trades (150 bars = ~3 days on 30m)
        can_trade = bars_since_last_trade >= 150
        
        if can_trade:
            # LONG ENTRIES
            # Require: 1d bullish + 4h bullish + Fisher long cross + volume + ADX
            # OR: Range regime + extreme Fisher + BB squeeze
            if trend_1d_bullish and trend_4h_bullish:
                if fisher_long_cross and volume_ok and adx_strong:
                    new_signal = current_size
            elif is_range and fisher_long_cross:
                if bb_squeeze and volume_ok:
                    new_signal = current_size * 0.7
            
            # SHORT ENTRIES
            # Require: 1d bearish + 4h bearish + Fisher short cross + volume + ADX
            # OR: Range regime + extreme Fisher + BB squeeze
            if trend_1d_bearish and trend_4h_bearish:
                if fisher_short_cross and volume_ok and adx_strong:
                    new_signal = -current_size
            elif is_range and fisher_short_cross:
                if bb_squeeze and volume_ok:
                    new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~6 days), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and trend_4h_bearish and fisher[i] > 1.0:
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
            if position_side > 0 and trend_1d_bearish and fisher[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and fisher[i] < -1.0:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher reaches +2.0 (overbought)
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher reaches -2.0 (oversold)
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or exits
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